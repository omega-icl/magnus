# This is the main makefile for MAGNUS
# Run `make help' for a list of available targets

export version = 1.1

export rootpath       = $(shell cd .. ; pwd)
export srcpath        = $(rootpath)/src
export interfacepath  = $(rootpath)/src/interface
export incpath        = $(rootpath)/include
export binpath        = $(rootpath)/bin
export libpath        = $(rootpath)/lib

#####

install: dirs_create interface_install magnus_install

dirs_create :
	@if test ! -e $(incpath)/; then \
		echo creating directory for header files; \
		mkdir $(incpath); \
	fi
	@if test ! -e $(libpath)/; then \
		echo creating directory for compiled libraries; \
		mkdir $(libpath); \
	fi

magnus_install:
	-(cd $(srcpath); make -f magnus.mk install)

interface_install:
	-(cd $(interfacepath); make -f magnus.mk install)

#####

clean: magnus_clean interface_clean

magnus_clean:
	-(cd $(srcpath); make -f magnus.mk clean)

interface_clean:
	-(cd $(interfacepath); make -f magnus.mk clean)

#####

uninstall: magnus_uninstall interface_uninstall

magnus_uninstall:
	-(cd $(srcpath); make -f magnus.mk uninstall)

interface_uninstall:
	-(cd $(interfacepath); make -f magnus.mk uninstall)

#####

help:
	@echo
	@echo "MAGNUS Makefile (ver. $(version))"
	@echo "Available targets:"
	@echo "  install     -  Create 'include', 'lib' and 'bin' directories, compile a shared library,"
	@echo "                 create symbolic links to header files and compiled libraries"
	@echo "  clean       -  Remove object files"
	@echo "  uninstall   -  Remove libraries, object files, and symbolic links"
	@echo "  help        -  Show this help message"
	@echo
