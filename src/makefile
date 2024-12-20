# This is the main makefile for MAGNUS
# - 'make install' creates the directories 'include', 'lib' and 'bin' (if not already
#   existing), compiles a shared library, and creates symbolic links to header files
#   and the compiled binaries and libraries
# - `make cleaninstall' removes the binaries, libraries, object files, and links

export version = 1.0

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

cleaninstall: canon_cleandist interface_cleandist

magnus_cleandist:
	-(cd $(srcpath); make -f magnus.mk cleandist)

interface_cleandist:
	-(cd $(interfacepath); make -f magnus.mk cleandist)

