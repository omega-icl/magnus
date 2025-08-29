# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_MAGNUS = $(shell cd ../.. ; pwd)
include $(PATH_MAGNUS)/src/makeoptions.mk

#####

all : test0 test1 test2

test% : test%.o
	$(CPP) $< $(LIB_MAGNUS) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $< -o $@

#####

clean:
	rm -f test? test??
	rm -f *.o *.out *.dat *.eps *.dot *.svg *.log *~

