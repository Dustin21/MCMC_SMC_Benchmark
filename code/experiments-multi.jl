using Distributions: Normal, Uniform, pdf
using PyPlot
include("MCMC.jl")
include("utils.jl")
include("SMCSampler.jl")
println("Loaded Libraries")

path = "/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/520\ -\ All\ about\ that\ bayes/BenchmarkingProject/Report/plots"

function f(x)
  m, n = size(x)
  y = 0.0
  y = x[:,1].^2
  for i = 2:n
    y = y + x[:,i].^2
  end
  return(mean(y))
end


function perform_experiment(sampler:: Function, iters:: Int64, particles:: Int64, function_list, true_values; seed_offset = 0, col_amount = 100)
  running_stats = zeros(n, length(function_list));
  errors = zeros(iters, int(n/col_amount), length(function_list));
  stable = zeros(iters, length(function_list));
  for i = 1:iters
    srand(seed_offset + i);
    x = sampler(particles);
    running_stats = means(x, function_list);
    errors[i, :, :] = collapsed_matrix(abs(broadcast(-, running_stats, true_values')), col_amount);
    for (j, val) in enumerate(true_values)
      stable[i, j] = stabalise(running_stats[:,j], val, 0.02)
    end
  end
  return stable, errors
end

function surfplot(fx, x1lim, x2lim; alpha = 1.0, n = 400, figsize = [10, 10], filename = "3dplot.png", path = "/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/520\ -\ All\ about\ that\ bayes/BenchmarkingProject/Report/plots")
  x1 = linspace(x1lim[1], x1lim[2], n)
  x2 = linspace(x2lim[1], x2lim[2],n)
  x1grid = repmat(x1',n,1)
  x2grid = repmat(x2,1,n)

  z = zeros(n,n)

  for i in 1:n
      for j in 1:n
          z[i,j] = fx([x1[i],x2[j]], alpha = alpha)
      end
  end
  #z = z./sum(z)
  plt.clf()
  fig = plt.figure("pyplot_surfaceplot",figsize=(figsize[1],figsize[2]))
  ax = fig[:add_subplot](2,1,1, projection = "3d")
  ax[:plot_surface](x1grid, x2grid, z, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("coolwarm"), alpha=0.8, linewidth=0.05)
  xlabel(L"$X_1$")
  ylabel(L"$X_2$")
  #zlim(0,0.0004)
  title("Multivariate Product Space")

  subplot(212)
  ax = fig[:add_subplot](2,1,2)
  cp = ax[:contour](x1grid, x2grid, z, cmap=ColorMap("coolwarm"), linewidth=0.5)
  ax[:clabel](cp, inline=1, fontsize=10)
  xlabel(L"$X_1$")
  ylabel(L"$X_2$")
  title("Contour Plot")
  tight_layout()
  savefig("$path/$filename")
end

# Function
function mix_gaussians(x, means = means, sds = sds, weights = weights, alpha = 1.0)
    l = length(means);
    out = 0;
    for i = 1:l
        out = out + weights[i] * exp(-(x + means[i]).^2 / (2*sds[i]))
    end
    return out.^(alpha).*exp(-(x).^2/2).^(1-alpha)
end

function print_equation(par, d, m, letters)
  for i = 1:d
    print("(")
    a = letters[i]
    for j = 1:m
      w = pars[i][3][j]
      u = pars[i][1][j]
      s = pars[i][2][j]
      print(@sprintf("%1.3f*exp(-(%s+%1.3f)^2/(2*%1.3f))", w, a, u, s))
      if j < m
        print("+")
      end
    end
    if(i < d)
      print(")*")
    else
      print(")\n")
    end
  end
end

function multi(x, pars; alpha = 1.0)
  d = length(x);
  out = 1.0;
  easy = 1.0;
  for i = 1:d
    m1 = pars[i][1];#10*rand(modes);
    sd1 = pars[i][2]; # 2*rand(modes) + 0.5;
    w1 = pars[i][3];# rand(modes) + 0.5;
    out = out .* mix_gaussians(x[i], m1, sd1, w1);
    easy = easy .* mix_gaussians(x[i], m1, sd1, w1, 0.0);
  end
  return (out).^alpha .* (easy).^(1-alpha)
end


function build_multi(d, modes = 5)
  pars = [(10*rand(modes), 2*rand(modes) + 0.5, rand(modes) + 0.5) for i = 1:d];
  m(x; alpha = 1.0) = multi(x, pars, alpha = alpha);
  return m, pars;
end


# Experiments
println("Started Experiments")
n = int(1e5); #number of samples
function_list = [f];
n_experiments = 1;
d = 5;

p(rand(10))
print(pars)

## Ground truth
srand(1)
p, pars = build_multi(2, 2)
mcmc_sampler(n) = mcmc_multi(p, rand(10), n = n, sig = 10);

#@time x = mcmc_sampler(int(1e7)) takes too long to run every time... uncomment to recalculate
#y107 = f(x)
#y106 = a
#println(y107)

print_equation(pars, 2, 2, ['y', 'x'])
surfplot(p, [-5, 5], [-5, 5])

fx = 106.467;

a = rand(5,10)
p(a[1,:])

#include("SMCSampler.jl")
n = int(1e5)
d= 2;
smc_sampler(n) = smcsampler_multi(p, rand(d), N=n, p = 5, σ = 15.0);
mcmc_sampler(n) = mcmc_multi(p, rand(d), n = n, sig = 10);
x, lw = smc_sampler(n)
x2 = mcmc_sampler(n)
f(x)
f(x2)

runs = 5

smc_x = zeros(Float64, runs, n, d);
mcmc_x = zeros(Float64, runs, n, d);
smc_time = zeros(Float64, runs);
mcmc_time = zeros(Float64, runs);
for i = 1:runs
  println(i)
  tic()
  smc_x[i, :, :] = smcsampler_multi(p, rand(d), N=n, p = 5, σ = 15.0);
  smc_time[i] = toc()
  tic()
  mcmc_x[i, :, :] = mcmc_sampler(n);
  mcmc_time[i] = toc()
end

mn = reshape(mean(smc_x, 3), runs, n)'

function fast_means(x::Array, f=default_f)
  n, d = size(x);
  out = zeros(Float64, n);
  out[1] = f(x[1, :]);
  for i = 2:n
    out[i] = (i-1)/i*out[i-1] + f(x[i, :])/i;
  end
  out
end

f(reshape(smc_x[2,:,:], n, d))

smc_y = zeros(Float64, runs, n)
mcmc_y = zeros(Float64, runs, n);
for i = 1:runs
  smc_y[i, :] = fast_means(reshape(smc_x[i,:,:], n, d), f);
  mcmc_y[i, :] = fast_means(reshape(mcmc_x[i,:,:], n, d), f);
end


smc_y_ave = mean(smc_y, 1)'
mcmc_y_ave = mean(mcmc_y, 1)'

writecsv("smc_y_ave.csv", smc_y_ave)
writecsv("mcmc_y_ave.csv", mcmc_y_ave)

writecsv("smc_y.csv", smc_y)
writecsv("mcmc_y.csv", mcmc_y)

# Plot figures
fig_x = 9;
fig_y = 5;

println("Plotting Figures 1/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
smc_plt = plt.plot(1:n, smc_y', color = cols[1]);
mcmc_plt = plt.plot(1:n,mcmc_y', color = cols[4]);
plt.title(L"Mean absolute error of $E[x]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Number of Particles", family = "serif");
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/E_X.png")
savefig("$path/E_X.pdf")

println("Plotting Figures 2/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt = plt.plot(smc_time, smc_er_mean[:,1], color = cols[1]);
mcmc_plt = plt.plot(mcmc_time, mcmc_er_mean[:,1], color = cols[4]);
plt.title(L"Mean absolute error of $E[x]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Average Walltime (s)", family = "serif");
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/E_X_walltime.png")
savefig("$path/E_X_walltime.pdf")

println("Plotting Figures 3/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt = plt.plot(1:n, smc_er_sd[:,1], color = cols[1]);
mcmc_plt = plt.plot(1:n, mcmc_er_sd[:,1], color = cols[4]);
plt.title(L"Standard Deviation of absolute error of $E[x]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Number of Particles", family = "serif");
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/sdE_X.png")
savefig("$path/sdE_X.pdf")

println("Plotting Figures 4/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt = plt.plot(1:n, smc_er_mean[:,2], color = cols[1]);
mcmc_plt = plt.plot(1:n, mcmc_er_mean[:,2], color = cols[4]);
plt.title(L"Mean absolute error of $E[x^2]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Number of Particles", family = "serif");
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/E_X2.png")
savefig("$path/E_X2.pdf")

println("Plotting Figures 5/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt = plt.plot(smc_time, smc_er_mean[:,2], color = cols[1]);
mcmc_plt = plt.plot(mcmc_time, mcmc_er_mean[:,2], color = cols[4]);
plt.title(L"Mean absolute error of $E[x^2]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Average Walltime (s)", family = "serif");
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/E_X2_walltime.png")
savefig("$path/E_X2_walltime.pdf")

println("Plotting Figures 6/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt = plt.plot(1:n, smc_er_sd[:,2], color = cols[1]);
mcmc_plt = plt.plot(1:n, mcmc_er_sd[:,2], color = cols[4]);
plt.title(L"Standard Deviation of absolute error of $E[x^2]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Number of Particles", family = "serif");
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/sdE_X2.png")
savefig("$path/sdE_X2.pdf")

println("Plotting Figures 7/8")
ex_errs = hcat(smc_stab[:,1],mcmc_stab[:,1])
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.figure(figsize=(fig_x,fig_y));
plt.hist(ex_errs, color = [cols[1], cols[4]], edgecolor = "none")
plt.title(L"Histogram of particles to within $2\%$ of true value of $E[x]$", family = "serif");
plt.ylabel("Frequency", family = "serif");
plt.xlabel("Number of Particles", family = "serif");
plt.legend(["SMC Sampler", "MCMC"],frameon = false, prop={"size"=>12, "family"=>"serif"})
plt.ylim(0,35)
savefig("$path/iterations.pdf")

println("Plotting Figures 8/8")
ex2_errs = hcat(smc_stab[:,2],mcmc_stab[:,2])
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.figure(figsize=(fig_x,fig_y));
plt.hist(ex2_errs, color = [cols[1], cols[4]], edgecolor = "none")
plt.title(L"Histogram of particles to within $2\%$ of true value of $E[x^2]$", family = "serif");
plt.ylabel("Frequency", family = "serif");
plt.xlabel("Number of Particles", family = "serif");
plt.legend(["SMC Sampler", "MCMC"],frameon = false, prop={"size"=>12, "family"=>"serif"})
plt.ylim(0,35)
savefig("$path/iterationsEx2.pdf")
