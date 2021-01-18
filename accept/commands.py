from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import click
import logging
import sys
import os
import subprocess
import json
import traceback
from .uncertain import umean
from collections import namedtuple
from . import core
from . import cwmemo


APPS = ['streamcluster', 'sobel', 'canneal', 'fluidanimate',
        'x264']
RESULTS_JSON = 'results.json'


GlobalConfig = namedtuple('GlobalConfig',
                          'client reps test_reps keep_sandboxes simulate')


@click.group(help='the ACCEPT approximate compiler driver')
@click.option('--verbose', '-v', count=True, default=0,
              help='log more output')
@click.option('--cluster', '-c', is_flag=True,
              help='execute on Slurm cluster')
@click.option('--force', '-f', is_flag=True,
              help='clear memoized results')
@click.option('--reps', '-r', type=int, default=1,
              help='replication factor')
@click.option('--test-reps', '-R', type=int, default=None,
              help='testing replication factor')
@click.option('--keep-sandboxes', '-k', is_flag=True,
              help='do not delete sandbox dirs')
@click.option('--simulate', '-s', is_flag=True,
              help='simulation (untrusted performance) mode')
@click.pass_context
def cli(ctx, verbose, cluster, force, reps, test_reps, keep_sandboxes,
        simulate):
    # Set up logging.
    logging.getLogger().addHandler(logging.StreamHandler(sys.stderr))
    if verbose >= 3:
        logging.getLogger().setLevel(core.FIREHOSE)
    elif verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    # Set up the parallelism/memoization client.
    client = cwmemo.get_client(cluster=cluster, force=force)

    # Testing reps fall back to training reps if unspecified.
    test_reps = test_reps or reps

    ctx.obj = GlobalConfig(client, reps, test_reps, keep_sandboxes, simulate)


# Utilities.

def get_eval(appdir, config):
    """Get an Evaluation object given the configured `GlobalConfig`.
    """
    return core.Evaluation(appdir, config.client, config.reps,
                           config.test_reps, config.simulate)


def dump_config(config):
    """Given a relaxation configuration and an accompanying description
    map, returning a human-readable string describing it.
    """
    optimizations = [r for r in config if r[1]]
    if not optimizations:
        return u'no optimizations'

    out = []
    for ident, param in optimizations:
        out.append(u'{} @ {}'.format(ident, param))
    return u', '.join(out)


def dump_result_human(res, verbose):
    """Dump a single Result object.
    """
    yield dump_config(res.config)
    if hasattr(res, 'error'):
        yield '{} % error'.format(res.error * 100)
    if hasattr(res, 'speedup'):
        yield '{} speedup'.format(res.speedup)
    if hasattr(res, 'duration'):
        yield '{} s duration'.format(res.duration)
    if verbose and hasattr(res, 'outputs'):
        output = res.outputs[0]
        if isinstance(output, basestring):
            yield 'output: {}'.format(output)
        elif isinstance(output, (list, tuple, dict)):
            if len(output) < 32:
                yield 'output: {}'.format(output)
            else:
                yield 'output is a {} of length {}'.format(
                    type(output).__name__, len(output)
                )
        elif output is not None:
            yield 'output has type {}'.format(type(output).__name__)
    if res.desc != 'good':
        yield res.desc


def dump_results_human(results, pout, verbose):
    """Generate human-readable text (as a sequence of lines) for
    the results.
    """
    optimal, suboptimal, bad = core.triage_results(results)

    if verbose and isinstance(pout, str):
        yield 'precise output: {}'.format(pout)
        yield ''

    yield '{} optimal, {} suboptimal, {} bad'.format(
        len(optimal), len(suboptimal), len(bad)
    )
    for res in optimal:
        for line in dump_result_human(res, verbose):
            yield line

    if verbose:
        yield '\nsuboptimal configs:'
        for res in suboptimal:
            for line in dump_result_human(res, verbose):
                yield line

        yield '\nbad configs:'
        for res in bad:
            for line in dump_result_human(res, verbose):
                yield line


def dump_results_json(results):
    """Return a JSON-like representation of the results.
    """
    results, _, _ = core.triage_results(results)
    out = []
    for res in results:
        out.append({
            'config': dump_config(res.config),
            'error_mu': res.error.value,
            'error_sigma': res.error.error,
            'speedup_mu': res.speedup.value,
            'speedup_sigma': res.speedup.error,
        })
    return out


# Run the paper experiments.

OPT_KINDS = {
    'loopperf': ('loop',),
    'desync':   ('lock', 'barrier'),
    'npu':      ('npu_region',),
}


def _triage_stats(results, test):
    prefix = 'test-' if test else 'train-'
    optimal, suboptimal, bad = core.triage_results(results)
    return {
        prefix + 'optimal': len(optimal),
        prefix + 'suboptimal': len(suboptimal),
        prefix + 'bad': len(bad),
    }


def run_experiments(ev, only=None, test=True):
    """Run all stages in the Evaluation for producing paper-ready
    results. Returns the main results, a dict of kind-restricted
    results, and elapsed time for the main results.
    """
    stats = {}

    # Main results.
    if only and 'main' not in only:
        main_results = []
        ev.setup()
    else:
        main_results, main_stats = ev.run()
    main_stats.update(_triage_stats(main_results, False))
    stats['main'] = main_stats

    # "Testing" phase.
    if test:
        main_results = ev.test_results(main_results)
        main_stats.update(_triage_stats(main_results, True))

    # Experiments with only one optimization type at a time.
    kind_results = {}
    for kind, words in OPT_KINDS.items():
        if only and kind not in only:
            continue

        # Filter all base configs for configs of this kind.
        logging.info('evaluating {} in isolation'.format(kind))
        kind_configs = []
        for config in ev.base_configs:
            for ident, param in config:
                if param and not ident.startswith(words):
                    break
            else:
                kind_configs.append(config)

        # Run the experiment workflow.
        logging.info('isolated configs: {}'.format(len(kind_configs)))
        train_results, kind_stats = ev.run(kind_configs)
        kind_stats.update(_triage_stats(train_results, False))
        if test:
            test_results = ev.test_results(train_results)
            kind_stats.update(_triage_stats(test_results, True))
            kind_results[kind] = test_results
        else:
            kind_results[kind] = train_results
        stats[kind] = kind_stats

    return main_results, kind_results, stats


@cli.command()
@click.argument('appdirs', metavar='DIR', nargs=-1,
                type=click.Path(file_okay=False, dir_okay=True, exists=True))
@click.option('--json', '-j', 'as_json', is_flag=True)
@click.option('--time', '-t', 'include_time', is_flag=True)
@click.option('--only', '-o', 'only', multiple=True)
@click.option('--verbose', '-v', is_flag=True,
              help='show suboptimal results')
@click.option('--notest', '-T', is_flag=True,
              help='disable test executions')
@click.pass_context
def exp(ctx, appdirs, verbose, as_json, include_time, only, notest):
    """Run experiments for the paper.
    """
    # Load the current results, if any.
    if as_json:
        try:
            with open(RESULTS_JSON) as f:
                results_json = json.load(f)
        except IOError:
            results_json = {}

    for appdir in appdirs:
        appname = os.path.basename(appdir)
        logging.info(appname)

        # Run the experiments themselves.
        exp = get_eval(appdir, ctx.obj)
        with exp.client:
            main_results, kind_results, stats = \
                run_experiments(exp, only, not notest)

        # Output as JSON to the results file.
        if as_json:
            out = {}
            out['main'] = dump_results_json(main_results)
            isolated = {}
            for kind, results in kind_results.items():
                isolated[kind] = dump_results_json(results)
            out['isolated'] = isolated
            if include_time:
                out['stats'] = stats

            if appname not in results_json:
                results_json[appname] = {}
            results_json[appname].update(out)

            with open(RESULTS_JSON, 'w') as f:
                json.dump(results_json, f, indent=2, sort_keys=True)

        # Output for human consumption.
        else:
            out = []
            if not only or 'main' in only:
                out += dump_results_human(main_results, exp.pout, verbose)
            if verbose or only:
                for kind, results in kind_results.items():
                    if only and kind not in only:
                        continue
                    out.append('')
                    if results:
                        out.append('ISOLATING {}:'.format(kind))
                        out += dump_results_human(results, exp.pout, verbose)
                    else:
                        out.append('No results for isolating {}.'.format(kind))
            print('\n'.join(out))


# Main ACCEPT workflow.

@cli.command()
@click.argument('appdir', default='.')
@click.option('--verbose', '-v', is_flag=True,
              help='show suboptimal results')
@click.option('--test', '-t', is_flag=True,
              help='test optimal configurations')
@click.pass_context
def run(ctx, appdir, verbose, test):
    """Run the ACCEPT workflow for a benchmark.

    Unlike the full experiments command (`accept exp`), this only gets
    the "headline" results for the benchmark; no characterization
    results for the paper are collected.
    """
    exp = get_eval(appdir, ctx.obj)

    with ctx.obj.client:
        results, _ = exp.run()

        # If we're getting test executions, run the optimal
        # configurations and instead print those results.
        if test:
            results = exp.test_results(results)

    print('base_time={} s'.format(umean(exp.ptimes)))
    pout = exp.test_pout if test else exp.pout
    output = dump_results_human(results, pout, verbose)
    for line in output:
        print(line)


# Get the compilation log or compiler output.

def log_and_output(directory, fn='accept_log.txt', keep=False):
    """Build the benchmark in `directory` and return the contents of the
    compilation log.
    """
    with core.chdir(directory):
        with core.sandbox(True, keep):
            if keep:
                logging.info('building in directory: {0}'.format(os.getcwd()))

            if os.path.exists(fn):
                os.remove(fn)

            output = core.build(require=False,
                                make_args=['OPTARGS=-accept-log'])

            if os.path.exists(fn):
                with open(fn) as f:
                    log = f.read()
            else:
                log = ''

            return log, output


@cli.command()
@click.argument('appdir', default='.')
@click.pass_context
def log(ctx, appdir):
    """Show ACCEPT optimization log.

    Compile the program---using the same memoized compilation as the
    `build` command---and show the resulting optimization log.
    """
    appdir = core.normpath(appdir)
    with ctx.obj.client:
        ctx.obj.client.submit(log_and_output, appdir,
                              keep=ctx.obj.keep_sandboxes)
        logtxt, _ = ctx.obj.client.get(log_and_output, appdir)

    # Pass the log file through c++filt.
    filtproc = subprocess.Popen(['c++filt'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
    out, _ = filtproc.communicate(logtxt)
    click.echo(out)


@cli.command()
@click.argument('appdir', default='.')
@click.pass_context
def build(ctx, appdir):
    """Compile a program and show compiler output.
    """
    appdir = core.normpath(appdir)
    with ctx.obj.client:
        ctx.obj.client.submit(log_and_output, appdir,
                              keep=ctx.obj.keep_sandboxes)
        _, output = ctx.obj.client.get(log_and_output, appdir)
    click.echo(output)


# Parts of the experiments.

@cli.command()
@click.argument('appdir', default='.')
@click.pass_context
def precise(ctx, appdir):
    """Execute the baseline version of a program.
    """
    ev = get_eval(appdir, ctx.obj)
    with ctx.obj.client:
        ev.setup()
        times = list(ev.precise_times())

    print('output:', ev.pout)
    print('time:')
    for t in times:
        print('  {:.2f}'.format(t))


@cli.command()
@click.argument('num', type=int, default=-1)
@click.argument('appdir', default='.')
@click.pass_context
def approx(ctx, num, appdir):
    """Execute approximate versions of a program.
    """
    ev = get_eval(appdir, ctx.obj)
    with ctx.obj.client:
        ev.run()
    results = ev.results

    # Possibly choose a specific result.
    results = [results[num]] if num != -1 else results

    for result in results:
        print(dump_config(result.config))
        print('output:')
        for output in result.outputs:
            print('  {}'.format(output))
        print('time:')
        for t in result.durations:
            if t is None:
                print('  (error)')
            else:
                print('  {:.2f}'.format(t))
        if result.desc != 'good':
            print(result.desc)
        print()


def main():
    try:
        cli()
    except core.UserError as exc:
        logging.debug(traceback.format_exc())
        logging.error(exc.log())
        sys.exit(1)


if __name__ == '__main__':
    main()
