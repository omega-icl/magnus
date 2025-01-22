# This makefile compiles a chared library of MAGNUS and it creates symbolic links
# to the header files in $(incpath), binaries in $(binpath), and libraries in $(libpath)

include $(srcpath)/makeoptions.mk

#####

incobjs = ffdoe.hpp base_mbdoe.hpp expdes.hpp modiscr.hpp
binobjs = 
libobjs = 

#binname = magnus
#libname = libmagnus.so

#####

install: dispBuild magnus_inc magnus_bin magnus_lib dispInstall
#	@if test ! -e $(binpath)/$(binname); then \
#		echo creating symolic link to executable $(binname); \
#		cd $(binpath) ; ln -s $(srcpath)/$(binname) $(binname); \
#	fi
#	@if test ! -e $(libpath)/$(libname); then \
#		echo creating symolic link to shared library $(libname); \
#		cd $(libpath) ; ln -s $(srcpath)/$(libname) $(libname); \
#	fi
	@echo

magnus_bin: $(binobjs)
#	$(CPP) $^ $(LIB_MAGNUS) -o $@ $(LDFLAGS)

magnus_lib: $(libobjs)
#	$(CPP) -shared -o $(libname) $(libobjs)

magnus_inc:
	@for INC in $(incobjs); do \
		if test ! -e $(incpath)/$$INC; then \
			echo creating symbolic link to header file $$INC; \
			cd $(incpath); ln -s $(srcpath)/$$INC $$INC; \
		fi; \
	done

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $< -o $@

%.o : %.c
	$(CPP) -c $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $< -o $@

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
	rm -fi $(libobjs) $(binobjs) $(binname) $(libname)

dispClean:
	@echo
	@(echo '***Cleaning MAGNUS directory (ver.' $(version)')***')
	@echo

#####

cleandist: dispCleanInstall
	rm -f $(libobjs) $(binname) $(libname)
	-(cd $(incpath) ; rm -f $(incobjs))
#	-(cd $(binpath) ; rm -f $(binname))
#	-(cd $(libpath) ; rm -f $(libname))
	
dispCleanInstall:
	@echo
	@(echo '***Uninstalling MAGNUS library (ver.' $(version)')***')
	@echo
