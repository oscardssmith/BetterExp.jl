module BetterExp

import Base.unsafe_trunc, Base.exponent_bias, Base.significand_bits, Base.sign_mask, Base.uinttype, Base.exponent_max

# magic rounding constant: 1.5*2^52 Adding, then subtracting it from a float rounds it to an Int.
MAGIC_ROUND_CONST(::Type{Float64}) = 6.755399441055744e15
MAGIC_ROUND_CONST(::Type{Float32}) = 1.048576f7

# min and max arguments by base and type
MAX_EXP(::Val{2},  ::Type{Float64}) =  1024.0                   # log2 2^1023*(2-2^-52)
MIN_EXP(::Val{2},  ::Type{Float64}) = -1075.0                   # log2 2^-1075
MAX_EXP(::Val{2},  ::Type{Float32}) =  128f0                    # log2 2^127*(2-2^-52)
MIN_EXP(::Val{2},  ::Type{Float32}) = -149f0                    # log2 2^-1075
MAX_EXP(::Val{ℯ},  ::Type{Float64}) =  709.7827128933845        # log 2^1023*(2-2^-52)
MIN_EXP(::Val{ℯ},  ::Type{Float64}) = -745.1332191019412076235  # log 2^-1075
MAX_EXP(::Val{ℯ},  ::Type{Float32}) =  88.72284f0               # log 2^127 *(2-2^-23)
MIN_EXP(::Val{ℯ},  ::Type{Float32}) = -103.97208f0              # log 2^-150
MAX_EXP(::Val{10}, ::Type{Float64}) =  308.25471555991675       # log10 2^1023*(2-2^-52)
MIN_EXP(::Val{10}, ::Type{Float64}) = -323.60724533877976       # log10 2^-1075
MAX_EXP(::Val{10}, ::Type{Float32}) =  38.53184f0               # log10 2^127 *(2-2^-23)
MIN_EXP(::Val{10}, ::Type{Float32}) = -45.1545f0                # log10 2^-150


# 256/log(base, 2) (For Float64 reductions)
LogBo256INV(::Val{2}, ::Type{Float64})    = 256.
LogBo256INV(::Val{ℯ}, ::Type{Float64})     = 369.3299304675746
LogBo256INV(::Val{10}, ::Type{Float64})   = 850.4135922911647
# -log(base, 2)/256 in upper and lower bits
LogBo256U(::Val{2}, ::Type{Float64})      = -0.00390625
LogBo256U(::Val{ℯ}, ::Type{Float64})       = -0.002707606173999011
LogBo256U(::Val{10}, ::Type{Float64})     = -0.0011758984205130218
LogBo256L(base::Val{2}, ::Type{Float64})  = 0.0
LogBo256L(base::Val{ℯ}, ::Type{Float64})  = -6.327543041662719e-14
LogBo256L(base::Val{10}, ::Type{Float64}) = -4.940469680332197e-14

# 1/log(base, 2) (For Float32 reductions)
LogBINV(::Val{2}, ::Type{Float32})    = 1f0
LogBINV(::Val{ℯ}, ::Type{Float32})    = 1.442695f0
LogBINV(::Val{10}, ::Type{Float32})   = 3.321928f0
# -log(base, 2) in upper and lower bits
LogBU(::Val{2}, ::Type{Float32})      = -1f0
LogBU(::Val{ℯ}, ::Type{Float32})      = -0.6931472f0 #-0.6931471805599453
LogBU(::Val{10}, ::Type{Float32})     = -0.30103f0
LogBL(base::Val{2}, ::Type{Float32})  =  0f0
LogBL(base::Val{ℯ}, ::Type{Float32})  = 1.9046542f-9
LogBL(base::Val{10}, ::Type{Float32}) = 1.4320989f-8

# Range reduced kernels          
@inline function expm1b_kernel(::Val{2}, x::Float64)
    return x * evalpoly(x, (0.6931471805599393, 0.24022650695910058,
                            0.05550411502333161, 0.009618129548366803))
end
@inline function expm1b_kernel(::Val{ℯ}, x::Float64)
    return x * evalpoly(x, (0.9999999999999912, 0.4999999999999997, 
                            0.1666666857598779, 0.04166666857598777))
end

@inline function expm1b_kernel(::Val{10}, x::Float64)
    return x * evalpoly(x, (2.3025850929940255, 2.6509490552391974, 
                            2.034678825384765, 1.1712552025835192))
end

@inline function expb_kernel(::Val{2}, x::Float32)
    return evalpoly(x, (1.0f0, 0.6931472f0, 0.2402265f0,
                        0.05550411f0, 0.009618025f0, 
                        0.0013333423f0, 0.00015469732f0, 1.5316464f-5))
end
@inline function expb_kernel(::Val{ℯ}, x::Float32)
    return evalpoly(x, (1.0f0, 1.0f0, 0.5f0, 0.16666667f0, 
                        0.041666217f0, 0.008333249f0,
                        0.001394858f0, 0.00019924171f0))
end
@inline function expb_kernel(::Val{10}, x::Float32)
    return evalpoly(x, (1.0f0, 2.3025851f0, 2.650949f0,
                        2.0346787f0, 1.1712426f0, 0.53937745f0, 
                        0.20788547f0, 0.06837386f0))
end


# Table stores data with 60 sig figs by using the fact that the first 12 bits of all the
# values would be the same if stored as regular Float64.
# This only gains 8 bits since the least significant 4 bits of the exponent
# of the small part are not the same for all table values.
const JU_MASK = 0x000FFFFFFFFFFFFF
const JU_CONST = 0x3ff0000000000000
const JL_CONST = 0x3c00000000000000
const J_TABLE= zeros(UInt64, 256);
for j in eachindex(J_TABLE)
    maskU = 0x000FFFFFFFFFFFFF
    maskL = 0x00FFFFFFFFFFFFFF
    val = 2.0^(big(j-1)/256)
    valU = Float64(val, RoundDown)
    valS = Float64(val-valU)
    valU = reinterpret(UInt64, valU) & maskU
    valS = ((reinterpret(UInt64, valS) & maskL)>>44)<<52
    J_TABLE[j] = valU | valS 
end
@inline function table_unpack(ind)
    j = @inbounds J_TABLE[ind]
    jU = reinterpret(Float64, JU_CONST | (j&JU_MASK))
    jL = reinterpret(Float64, JL_CONST | (j>>8))
    return jU, jL
end

# Method
# 1. Argument reduction: Reduce x to an r so that |r| <= log(2)/512. Given x,
#    find r and integers k, j such that
#       x = (k + j/256)*log(2) + r,  0 <= j < 256, |r| <= log(2)/512.
#
# 2. Approximate exp(r) by it's degree 3 taylor series around 0.
#    Since the bounds on r are very tight, this is sufficient to be accurate to floating point epsilon.
#
# 3. Scale back: exp(x) = 2^k * 2^(j/256) * exp(r)
#    Since the range of possible j is small, 2^(j/256) is simply stored for all possible values.
for (func, base) in (:exp2=>Val(2), :exp=>Val(ℯ), :exp10=>Val(10))
    @eval begin
        @inline function ($func)(x::T) where T<:Float64
            N_float = muladd(x, LogBo256INV($base, T), MAGIC_ROUND_CONST(T))
            N = reinterpret(uinttype(T), N_float) % Int32
            N_float -=  MAGIC_ROUND_CONST(T)
            r = muladd(N_float, LogBo256U($base, T), x)
            r = muladd(N_float, LogBo256L($base, T), r)
            k = N >> 8
            jU, jL = table_unpack(N&255 + 1)
            small_part =  muladd(jU, expm1b_kernel($base, r), jL) + jU
            
            if !(abs(k) <= 53)
                isnan(x) && return x
                x >= MAX_EXP($base, T) && return Inf
                x <= MIN_EXP($base, T) && return 0.0
                if k <= -53
                    twopk = (k + 53) << 52 
                    return reinterpret(T, twopk + reinterpret(UInt64, small_part))*(2.0^-53)
                end
            end
            twopk = Int64(k) << 52
            return reinterpret(T, twopk + reinterpret(Int64, small_part))
        end
    end
    
    @eval begin
        @inline function ($func)(x::T) where T<:Float32
            N_float = round(x*LogBINV($base, T))
            N = unsafe_trunc(Int32, N_float)
            r = muladd(N_float, LogBU($base, T), x)
            r = muladd(N_float, LogBL($base, T), r)
            small_part = expb_kernel($base, r)
            if !(abs(N)<=Int32(24))
                isnan(x) && return x
                x > MAX_EXP($base, T) && return Inf32
                x < MIN_EXP($base, T) && return 0.0f0
                if N<=Int32(-24)
                    twopk = reinterpret(T, (N+Int32(151)) << Int32(23))
                    return (twopk*small_part)*(2f0^(-24))
                end
            end
            twopk = reinterpret(T, (N+Int32(127)) << Int32(23))
            return twopk*small_part
        end
    end
end
end # module
