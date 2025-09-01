# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_MAGNUS = $(shell cd ../.. ; pwd)
include $(PATH_MAGNUS)/src/makeoptions.mk

#####

all : test0 test1 test1b test1c test2 test3 test4 test5 test6 test7 test7b test8

test% : test%.o
	$(CPP) $< $(LIB_MAGNUS) -o $@

%.o : %.cpp
	$(CPP) -c $(FLAG_CPP) $(FLAG_MAGNUS) $(INC_MAGNUS) $< -o $@

#####

clean:
	rm -f test? test?b
	rm -f *.o *.out *.dat *.eps *.dot *.svg *.log *~

