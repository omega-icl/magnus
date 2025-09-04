# This makefile compiles a shared library of MAGNUS and it creates symbolic links
# to the header files in $(incpath), binaries in $(binpath), and libraries in $(libpath)

include $(srcpath)/makeoptions.mk

#####

incobjs = base_sampling.hpp \
          fffeas.hpp base_mbfa.hpp nsfeas.hpp \
          ffest.hpp base_parest.hpp parest.hpp \
          ffdoe.hpp base_mbdoe.hpp expdes.hpp modiscr.hpp
binobjs =
libobjs =

#binname = magnus
#libname = libmagnus.so

#####

install: dispBuild magnus_inc magnus_bin magnus_lib dispInstall
#	@if test ! -e $(binpath)/$(binname); then \
#		echo creating symbolic link to executable $(binname); \
#		cd $(binpath); ln -s $(srcpath)/$(binname) $(binname); \
#	fi
#	@if test ! -e $(libpath)/$(libname); then \
#		echo creating symbolic link to shared library $(libname); \
#		cd $(libpath); ln -s $(srcpath)/$(libname) $(libname); \
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

%.o: %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $< -o $@

%.o: %.c
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

uninstall: dispUninstall
	rm -f $(libobjs) $(binobjs) $(binname) $(libname)
	-(cd $(incpath); rm -f $(incobjs))
#	-(cd $(binpath); rm -f $(binname))
#	-(cd $(libpath); rm -f $(libname))

dispUninstall:
	@echo
	@(echo '***Uninstalling MAGNUS library (ver.' $(version)')***')
	@echo
