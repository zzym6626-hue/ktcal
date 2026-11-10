CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Werror -pthread

.PHONY: test board clean
test:
	$(CXX) $(CXXFLAGS) fast_path.cc test_fast_path.cc -o ktcal_fast_test
	./ktcal_fast_test

# Rockchip RKNN Runtime C SDK (and matching target sysroot) is supplied by user.
board:
	@test -n "$(RKNN_INC)" -a -n "$(RKNN_LIB)" || (echo 'set RKNN_INC and RKNN_LIB' >&2; exit 1)
	$(CXX) $(CXXFLAGS) -I$(RKNN_INC) fast_path.cc rknn_backend.cc main.cc \
	    -L$(RKNN_LIB) -lrknnrt -o ktcal_rv1106

clean:
	rm -f ktcal_fast_test ktcal_rv1106
