using Distributions: Normal
function mcmc(p::Function, x0; sig = 0.6, n = 20000, stype = Float64)
  #intitialize the sampling. Start somewhere from 0..1
  x = zeros(Float64, n); # to save our samples
  N = length(x0);
  x[1] = x0;
  u = 0.0;
  x_star = 0.0;
  for i = 2:n
    # sample new state candidate from proposal distribution
    for j = 1:N
      x_star = x[i-1, j] + rand(Normal(0.0,sig));
    end
    u = rand(); #coin flip
    # A is simple because our transition probability is symmetric (Metropolis only)
    if u < p(x_star)/p(x[i-1]) #coin flip to see if we accept
      x[i] = x_star;
    else
      x[i] = x[i-1];
    end
  end
  return x
end



function mcmc_multi(p::Function, x0; sig = 0.6, n = 20000, stype = Float64)
  #intitialize the sampling. Start somewhere from 0..1
  d = length(x0);
  x = zeros(Float64, n, d); # to save our samples
  x[1, :] = x0;
  u = 0.0;
  x_star = zeros(Float64, 1, d);
  for i = 2:n
    # sample new state candidate from proposal distribution
    x_star = x[i, :] + rand(Normal(0.0,sig), (1, d));
    u = rand(); #coin flip
    # A is simple because our transition probability is symmetric (Metropolis only)
    if u < p(x_star)/p(x[i-1,:]) #coin flip to see if we accept
      x[i,:] = x_star;
    else
      x[i,:] = x[i-1, :];
    end
  end
  return x
end

