
# magic rounding constant: 1.5*2^52 Adding, then subtracting it from a float rounds it to an Int.
MAGIC_ROUND_CONST(::Type{Float64}) = 6.755399441055744e15
MAGIC_ROUND_CONST(::Type{Float32}) = 1.048576f7

@generated MAX_EXP(n::Val{N}, ::Type{T}) where {N, T} = T(exponent_bias(T)*log(N, big(2)) + log(N, 2 - big(2.0)^-significand_bits(T)))
@generated MIN_EXP(n::Val{N}, ::Type{T}) where {N, T} = T(-(exponent_bias(T)+significand_bits(T)) * log(N, big(2)))
@generated SUBNORM_EXP(n::Val{N}, ::Type{T}) where {N, T} = log(N, floatmin(T))


# 256/log(base, 2) (For Float64 reductions)
@generated LogBo256INV(::Val{N}, ::Type{Float64}) where {N} = Float64(256/log(N, big(2)))

# -log(base, 2)/256 in upper and lower bits
# Upper is truncated to only have 34 bits of significand since N has at most
# ceil(log2(-MIN_EXP(n, Float64)*LogBo256INV(Val(2), Float64))) = 19 bits.
# This ensures no rounding when multiplying LogBo256U*N for FMAless hardware
@generated function LogBo256U(::Val{N}, ::Type{Float64}) where {N} 
    val = reinterpret(UInt64, -Float64(log(N, big(2)))/256)
    return reinterpret(Float64, val & typemax(UInt64)<<19)
end
@generated function LogBo256L(::Val{N}, ::Type{Float64}) where {N}
    return Float64(-log(N, big(2))/256-LogBo256U(Val(N), Float64))
end

# 1/log(base, 2) (For Float32 reductions)
@generated LogBINV(::Val{N}, ::Type{Float32}) where {N} = Float32(1/log(N, big(2)))

# -log(base, 2) in upper and lower bits
# Upper is truncated to only have 16 bits of significand since N has at most
# ceil(log2(-MIN_EXP(n, Float32)*LogBoINV(Val(2), Float32))) = 8 bits.
# This ensures no rounding when multiplying LogBU*N for FMAless hardware
@generated function LogBU(::Val{N}, ::Type{Float32}) where {N} 
    val = reinterpret(UInt32, -Float32(log(N, big(2))))
    return reinterpret(Float32, val & typemax(UInt32)<<8)
end
@generated function LogBL(::Val{N}, ::Type{Float32}) where {N}
    return Float32(-log(N, big(2))-LogBU(Val(N), Float32))
end

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
# of the small part are not the same for all ta
const JU_MASK = typemax(UInt64)>>12
const JL_MASK = typemax(UInt64)>>8
const JU_CONST = 0x3FF0000000000000
const JL_CONST = 0x3C00000000000000
const J_TABLE= zeros(UInt64, 256);
for j in eachindex(J_TABLE)
    val = 2.0^(big(j-1)/256)
    valU = Float64(val, RoundDown)
    valS = Float64(val-valU)
    valU = reinterpret(UInt64, valU) & JU_MASK
    valS = ((reinterpret(UInt64, valS) & JL_MASK)>>44)<<52
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
            
            if !(abs(x) <= SUBNORM_EXP($base, T))
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
        
        @inline function ($func)(x::T) where T<:Float32
            N_float = round(x*LogBINV($base, T))
            N = unsafe_trunc(Int32, N_float)
            r = muladd(N_float, LogBU($base, T), x)
            r = muladd(N_float, LogBL($base, T), r)
            small_part = expb_kernel($base, r)
            if !(abs(x) <= SUBNORM_EXP($base, T))
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
