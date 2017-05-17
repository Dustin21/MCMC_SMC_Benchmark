using Distributions: Normal, Multinomial, Binomial, MvNormal, pdf
# Useful functions
ess(w) = 1/sum(exp(2*w));
phi(n, p) = n/p;

function resample_idx(p:: Array{Float64}, idx::Array{Int32})
  # Fairly fast function for doing resampling
  n = length(p);
  y = rand(Multinomial(n, p));
  k = 1;
  n_part = 0;
  for i in 1:n
    n_part = y[i];
    for j = 1:n_part
      idx[k] = i;
      k = k+1;
    end
  end
  idx;
end

function reset_weights!(X::Array{Float64}, value::Float64, idx::Int64)
  # Resets the weights in a row of a matrix to the value given in value. Operates in place
  for i = 1:length(X[idx,:])
    X[idx, i] = value
  end
end

function reset_weights_multi!(X::Array{Float64}, value::Float64)
  # Resets the weights in a row of a matrix to the value given in value. Operates in place
  for i = 1:length(X)
    X[i] = value
  end
end

function normalise(x::Array)
  mx = maximum(x);
  z = sum(exp(x .- mx));
  return log(exp(x .- mx)./z);
end

function weight_update!(X::Array{Float64}, value::Float64, idx::Int64)
  # Resets the weights in a row of a matrix to the value given in value. Operates in place
  for i = 1:length(X[idx,:])
    X[idx, i] = value
  end
end

function smcsampler(γ::Function; δ = 0.5, N = 1000, p = 100, σ = 0.5)
  # γ = ESS threshold
  # N = number of particles
  # p = Annealing steps π_n(x) ∝ [π(x)]^ϕ_n [μ_1(x)]^(1-ϕ_n)   0 ≤ ϕ_1 ≤ ... ≤ ϕ_p = 1
  # δ : N*δ = T, the threshold at which we resample the particles

  q = Normal(0, σ); # Proposal Distribution
  xs = zeros(Float64, p, N); # particles
  lw = zeros(Float64, p, N); # weights
  essi = zeros(Float64, p); # ESS values
  # SMC Algorithm
  # INITIALIZATION
  n = 1;
  # sample particles initial distribution
  xs[n, :] = rand(q, N);
  lw[n, :] = log(γ(xs[n, :], 0.0)) - log(pdf(q, xs[n, :]));
  lw[n, :] = normalise(lw[n, :]);
  idx = zeros(Int32, N);
  while true
    # RESAMPLING
    essi[n]=ess(lw[n,:]);
    if essi[n] < N*δ
      idx = resample_idx(reshape(exp(lw[n, :]),N), idx);
      xs[n, :] = xs[n, idx];
      reset_weights!(lw, log(1/N), n);
    end
    n = n + 1;
    if n == p+1
      break;
    end

    xs[n, :] = xs[n - 1, :] + reshape(rand(q,N), 1, N);
    lw[n, :] = lw[n-1] + log(γ(xs[n, :], 1.0*n/p)) - log(pdf(q, xs[n, :] - xs[n-1, :]));
    lw[n, :] = normalise(lw[n, :]);
  end

  #return xs, lw, essi;

  return xs;
end

function smcsampler_multi(γ::Function, x0; δ = 0.5, N = 1000, p = 100, σ = 0.5)
  # γ = ESS threshold
  # N = number of particles
  # p = Annealing steps π_n(x) ∝ [π(x)]^ϕ_n [μ_1(x)]^(1-ϕ_n)   0 ≤ ϕ_1 ≤ ... ≤ ϕ_p = 1
  # δ : N*δ = T, the threshold at which we resample the particles
  d = length(x0);
  q = MvNormal(σ*eye(d)); # Proposal Distribution
  x = zeros(Float64, N, d); # particles
  x_prev = zeros(Float64, N, d); # particles
  lw = zeros(Float64, N); # weights
  lw_prev = zeros(Float64, N); # weights
  essi = 0.0; # ESS values
  # SMC Algorithm
  # INITIALIZATION
  n = 1;
  # sample particles initial distribution
  x = rand(q,N)';
  for i = 1:N
    lw[i] = log(γ(x[i, :], alpha = 0.0)) - log(pdf(q, reshape(x[i, :],d)));
  end
  lw = normalise(lw);
  idx = zeros(Int32, N);
  while true
    # RESAMPLING
    essi = ess(lw);
    if essi < N*δ
      idx = resample_idx(reshape(exp(lw),N), idx);
      x = x[idx, :];
      reset_weights_multi!(lw, log(1/N));
    end
    x_prev = x;
    lw_prev = lw;
    n = n + 1;
    if n == p+1
      break;
    end

    x = x_prev + rand(q,N)';
    for i = 1:N
      lw[i] = lw_prev[i] + log(γ(x[i, :], alpha = 1.0*n/p)) - log(pdf(q, reshape(x[i, :] - x_prev[i, :],d)));
    end
    #lw[n, :] = lw[n-1] + log(γ(xs[n, :], 1.0*n/p)) - log(pdf(q, xs[n, :] - xs[n-1, :]));
    lw = normalise(lw);
  end
  #idx = resample_idx(reshape(exp(lw),N), idx);
  #x = x[idx, :];
  return x;
end

