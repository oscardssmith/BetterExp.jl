module BetterExp

import Base.unsafe_trunc, Base.exponent_bias, Base.significand_bits, Base.sign_mask, Base.uinttype, Base.exponent_max

#using Base: exp, exp2, exp10
#using Base.Fastmath: fast_exp, fast_exp2, fast_exp10

include("exp.jl")
#include("expm1.jl")
end
