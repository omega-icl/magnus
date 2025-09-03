# THIRD-PARTY LIBRARIES <<-- CHANGE AS APPROPRIATE -->>

PATH_MAGNUS_MK := $(dir $(lastword $(MAKEFILE_LIST)))

include $(abspath $(PATH_MAGNUS_MK)../../canon/src/makeoptions.mk)
PATH_CANON = $(abspath $(PATH_MAGNUS_MK)../../canon)

PATH_MAGNUS = $(abspath $(PATH_MAGNUS_MK)../)

# COMPILATION <<-- CHANGE AS APPROPRIATE -->>

PROF = #-pg
OPTIM = -O2
DEBUG = #-g
WARN  = -Wall -Wno-misleading-indentation -Wno-unknown-pragmas -Wno-parentheses -Wno-unused-result
CPP17 = -std=c++17
CC    = gcc
CPP   = g++
# CPP   = icpc

# <<-- NO CHANGE BEYOND THIS POINT -->>

FLAG_CPP  = $(DEBUG) $(OPTIM) $(CPP17) $(WARN) $(PROF)
LINK      = $(CPP)
FLAG_LINK = $(PROF)

FLAG_MAGNUS = $(FLAG_CANON)
LIB_MAGNUS  = $(LIB_CANON)
INC_MAGNUS  = -I$(PATH_MAGNUS)/src $(INC_CANON)

