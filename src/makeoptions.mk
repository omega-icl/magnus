# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_MAGNUS    = $(shell cd $(HOME)/Programs/bitbucket/magnus; pwd)
PATH_CANON     = $(shell cd $(HOME)/Programs/bitbucket/canon40; pwd)

# COMPILATION <<-- CHANGE AS APPROPRIATE -->>

include $(PATH_CANON)/src/makeoptions.mk

PROF = #-pg
OPTIM = #-O2
DEBUG = -g
WARN  = -Wall -Wno-misleading-indentation -Wno-unknown-pragmas -Wno-parentheses -Wno-unused-result
CPP17 = -std=c++17
CC    = gcc-13
CPP   = g++-13
# CPP   = icpc

# <<-- NO CHANGE BEYOND THIS POINT -->>

FLAG_CPP  = $(DEBUG) $(OPTIM) $(CPP17) $(WARN) $(PROF)
LINK      = $(CPP)
FLAG_LINK = $(PROF)

FLAG_MAGNUS = $(FLAG_CANON)
LIB_MAGNUS  = $(LIB_CANON)
INC_MAGNUS  = -I$(PATH_MAGNUS)/src $(INC_CANON)

