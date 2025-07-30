# This makefile compiles the python interface and creates a symbolic link
# to the libray in $(libpath)

include $(srcpath)/makeoptions.mk

#####

libobjs   = nsfeas.o parest.o magnus.o
libname   = magnus.so
libdep    = pymc.so cronos.so canon.so

#####

install: dispBuild $(libname) dispInstall
	@if test ! -e $(libpath)/$(libname); then \
		echo creating symolic link to shared library $(libname); \
		cd $(libpath) ; ln -s $(interfacepath)/$(libname) $(libname); \
	fi
	@for DEP in $(libdep); do \
		echo dependent library $$DEP; \
		if test ! -e $$DEP; then \
			ln -s $(PATH_CANON)/lib/$$DEP; \
		fi; \
		if test ! -e $(libpath)/$$DEP; then \
			echo creating symolic link to shared library $$DEP; \
			cd $(libpath) ; ln -s $(PATH_CANON)/lib/$$DEP; \
		fi; \
	done
	@echo

$(libname): $(libobjs)
	$(CPP) -shared -Wl,--export-dynamic $(libobjs) $(LIB_MAGNUS) -o $(libname) 

%.o : %.cpp
	$(CPP) $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $(INC_PYBIND11) -c $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $< -o $@

%.c : $(PATH_GAMS)/apifiles/C/api/%.c
	cp $< $@

dispBuild:
	@echo
	@(echo '***Compiling MAGNUS library (ver.' $(version)')***')
	@echo

dispInstall:
	@echo
	@(echo '***Installing MAGNUS library (ver.' $(version)')***')
	@echo

#####

clean: dispClean
	rm -fi $(libobjs) $(libname) $(libdep)

dispClean:
	@echo
	@(echo '***Cleaning MAGNUS directory (ver.' $(version)')***')
	@echo

#####

cleandist: dispCleanInstall
	rm -f $(libobjs) $(libname) $(libdep)
	-(cd $(libpath) ; rm -f $(libname) $(libdep))

dispCleanInstall:
	@echo
	@(echo '***Uninstalling MAGNUS library (ver.' $(version)')***')
	@echo

