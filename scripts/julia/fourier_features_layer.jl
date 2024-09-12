struct FourierFeatures <: Lux.AbstractExplicitLayer
    indims::Int
    outdims::Int
end
Lux.initialparameters(rng::AbstractRNG, layer::FourierFeatures) = (;)
Lux.initialstates(rng::AbstractRNG, layer::FourierFeatures) = (
    B = 2π.*randn(Int(layer.outdims//2), layer.indims)*0.01,
)
(ff::FourierFeatures)(x, p, st) = vcat(cos.(st.B*x), sin.(st.B*x)), st
