CXX ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Werror -pthread
# Official RV1106/1103 SDK uses the mini runtime; override for a different SDK.
RKNN_RUNTIME ?= rknnmrt

.PHONY: test board board-causal16 clean
test:
	$(CXX) $(CXXFLAGS) fast_path.cc test_fast_path.cc -o ktcal_fast_test
	./ktcal_fast_test
	$(CXX) $(CXXFLAGS) -I../../tests/fixtures/rknn fast_path.cc sequence_path.cc \
	    rknn_sequence_backend.cc test_sequence_backend.cc -o /tmp/ktcal_sequence_backend_test
	/tmp/ktcal_sequence_backend_test

# Rockchip RKNN Runtime C SDK (and matching target sysroot) is supplied by user.
board:
	@test -n "$(RKNN_INC)" -a -n "$(RKNN_LIB)" || (echo 'set RKNN_INC and RKNN_LIB' >&2; exit 1)
	$(CXX) $(CXXFLAGS) -I$(RKNN_INC) fast_path.cc sequence_path.cc rknn_sequence_backend.cc sequence_main.cc \
	    -L$(RKNN_LIB) -l$(RKNN_RUNTIME) -o ktcal_rv1106

board-causal16:
	@test -n "$(RKNN_INC)" -a -n "$(RKNN_LIB)" || (echo 'set RKNN_INC and RKNN_LIB' >&2; exit 1)
	$(CXX) $(CXXFLAGS) -I$(RKNN_INC) fast_path.cc rknn_backend.cc main.cc \
	    -L$(RKNN_LIB) -l$(RKNN_RUNTIME) -o ktcal_causal16

clean:
	rm -f ktcal_fast_test ktcal_rv1106 ktcal_causal16
