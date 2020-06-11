BUILD := build
BUILT := $(BUILD)/built
LLVM_VERSION := 3.2

# Program names.
CMAKE := cmake
VIRTUALENV := virtualenv

# Location of the Python virtual environment.
VENV := venv

CXXLIBPATH := $(wildcard /usr/include/*-linux-gnu/c++/4.*)
ARCH_CXXLIB:= $(shell basename /usr/include/*-linux-gnu/)
TOOLCHAIN_VERSION := $(shell basename $(CXXLIBPATH))
TOOLCHAIN_FOLDER := gcc_toolchain

# CMake options for building LLVM and the ACCEPT pass.
CMAKE_FLAGS := -G Ninja -DCMAKE_INSTALL_PREFIX:PATH=$(shell pwd)/$(BUILT) 
ifeq ($(RELEASE),1)
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE:STRING=Release
else
CMAKE_FLAGS += -DCMAKE_BUILD_TYPE:STRING=Debug
endif

ifeq ($(shell uname -s),Darwin)
        LIBEXT := dylib
else
        LIBEXT := so
endif

# Automatically use a binary called "ninja-build", if it's available. Some
# package managers call it this to avoid naming conflicts.
ifeq ($(shell which ninja-build >/dev/null 2>&1 ; echo $$?),0)
	NINJA := ninja-build
else
	NINJA := ninja
endif

# LLVM 3.2 has some trouble building against libc++, which seems to be the
# default standard library on recent OS X dev tools. Presumably this is fixed
# in later versions of LLVM, but for now, we force the compiler to use GNU
# libstdc++.
ifeq ($(LLVM_VERSION),3.2)
ifneq ($(shell c++ --version | grep clang),)
	CMAKE_FLAGS += '-DCMAKE_CXX_FLAGS:STRING=-stdlib=libstdc++ -std=gnu++98'
endif
endif

# On platforms that ship Python 3 as `python`, force Python 2 to be used in
# CMake. I don't think CMake itself has a problem with py3k, but LLVM's
# scripts do.
# Additionally, make virtualenv use python2.
ifeq ($(shell which python2 >/dev/null 2>&1 ; echo $$?),0)
	PYTHON2 := python2
	CMAKE_FLAGS += -DPYTHON_EXECUTABLE:PATH=$(shell which python2)
	VIRTUALENV += -p $(shell which python2)
else
	PYTHON2 := python
endif
# If python2-virtualenv is installed, use that instead.
ifeq ($(shell which virtualenv2 >/dev/null 2>&1 ; echo $$?),0)
	VIRTUALENV := virtualenv2
endif


# Actually building stuff.

.PHONY: accept llvm

accept: check_cmake check_ninja
	mkdir -p $(BUILD)/enerc
	cd $(BUILD)/enerc ; $(CMAKE) $(CMAKE_FLAGS) ../..
	cd $(BUILD)/enerc ; $(NINJA) install

llvm: llvm/CMakeLists.txt llvm/tools/clang check_cmake check_ninja
	# To prevent clang from using gcc newer toolchain (and fail compiling),
	# create our own toolchain
	if [ "$(CXXLIBPATH)" != "" ]; then \
		mkdir $(TOOLCHAIN_FOLDER); \
		cd $(TOOLCHAIN_FOLDER); \
		ln -s /usr/include include; \
		ln -s /usr/bin bin; \
		mkdir -p lib/gcc/$(ARCH_CXXLIB); \
		cd lib/gcc/$(ARCH_CXXLIB)/; \
		ln -s /usr/lib/gcc/$(ARCH_CXXLIB)/$(TOOLCHAIN_VERSION) $(TOOLCHAIN_VERSION); \
	fi
	# Actually building llvm
	mkdir -p $(BUILD)/llvm
	cd $(BUILD)/llvm ; $(CMAKE) $(CMAKE_FLAGS) ../../llvm
	cd $(BUILD)/llvm ; $(NINJA) install


# Convenience targets.

.PHONY: setup test clean

setup: llvm accept driver
	ln -s $(BUILD)/enerc/compile_commands.json

test:
	$(PYTHON2) $(BUILT)/bin/llvm-lit -v --filter='test_\w+\.' test

clean:
	rm -rf $(BUILD)
	rm -rf $(TOOLCHAIN_FOLDER)
	rm compile_commands.json

# Fetching and extracting LLVM.

.INTERMEDIATE: llvm-$(LLVM_VERSION).src.tar.gz
llvm-$(LLVM_VERSION).src.tar.gz:
	curl -LO http://releases.llvm.org/$(LLVM_VERSION)/$@

llvm/CMakeLists.txt: llvm-$(LLVM_VERSION).src.tar.gz
	tar -xf $<
	mv llvm-$(LLVM_VERSION).src llvm

# Symlink our modified Clang source into the LLVM tree. This way, building the
# "llvm" directory will build both LLVM and Clang. (In fact, this is the only
# way to build Clang at all as far as I know.)
llvm/tools/clang: llvm/CMakeLists.txt
	cd llvm/tools ; ln -s ../../clang .


# Friendly error messages when tools don't exist.

.PHONY: check_cmake check_ninja check_virtualenv

check_cmake:
	@if ! $(CMAKE) --version > /dev/null ; then \
		echo "Please install CMake to build LLVM and ACCEPT."; \
		echo "http://www.cmake.org"; \
		exit 2; \
	else true; fi

check_ninja:
	@if ! $(NINJA) --version > /dev/null ; then \
		echo "Please install Ninja to build LLVM and ACCEPT."; \
		echo "http://martine.github.io/ninja/"; \
		exit 2; \
	else true; fi

check_virtualenv:
	@if ! $(VIRTUALENV) --version > /dev/null ; then \
		echo "Please install Virtualenv to use the ACCEPT driver."; \
		echo "http://www.virtualenv.org/"; \
		exit 2; \
	else true; fi


# Python driver installation.

.PHONY: driver

# Make a virtualenv and install all our dependencies there. This avoids
# needing to clutter the system Python libraries (and possibly requiring
# sudo).
driver:
	[ -e $(VENV)/bin/pip ] || $(VIRTUALENV) $(VENV)
	./$(VENV)/bin/pip install -r requirements.txt


# Documentation.

.PHONY: docs cleandocs deploy

docs:
	mkdocs build

cleandocs:
	rm -rf site

# Upload the documentation to the Web server.
CSEHOST := bicycle.cs.washington.edu
CSEPATH := /cse/www2/sampa/accept
deploy: cleandocs docs
	rsync --compress --recursive --checksum --itemize-changes --delete -e ssh site/ $(CSEHOST):$(CSEPATH)
