using Distributions: Normal, Uniform, pdf
using PyPlot
include("MCMC.jl")
include("utils.jl")
include("SMCSampler.jl")
println("Loaded Libraries")
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
cols = Tuple[]
for i = 1:20
  push!(cols, (tableau20[i][1]/255.0, tableau20[i][2]/255.0, tableau20[i][3]/255.0));
end
cols

function means(x, functions = [g])
  summaries = zeros(Float64, length(x), length(functions));
  for (i, fun) in enumerate(functions)
    summaries[:, i] = fast_means(x, fun);
  end
  return(summaries)
end

function write_matrix(f::IOStream, data::Array, exp)
  m, n = size(data);
  for i in 1:m
    write(f, string(exp));
    for j in 1:n
      write(f, ", ");
      write(f, string(data[i,j]));
    end
    write(f, "\n");
  end
end

path = "/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/520\ -\ All\ about\ that\ bayes/BenchmarkingProject/Report/plots"

function collapsed_matrix(x, amount = 100)
  m,n = size(x);
  d = int(m/amount)
  collapse = zeros(Float64, d, n);
  for i = 1:d
    collapse[i,:] = mean(x[(i-1)*amount + 1: (i)*amount, :], 1)
  end
  return collapse
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

# Function
p(x, alpha = 1.0) = (exp(-(x - 2).^2/2) +
                       0.5*exp(-(x+2).^2/1) +
                       0.5*exp(-(x-5).^2/1) +
                       0.5*exp(-(x-15).^2/1)).^(alpha).*exp(-(x).^2/2).^(1-alpha);
# Ground truth
e_x = 4.05887;
e_x2 = 46.2633;
true_values = [e_x; e_x2];

# Experiments
println("Started Experiments")
n = int(1e6); #number of samples
g(x) = x.^2; f(x) = x; function_list = [f, g];
n_experiments = 50;
d = 5;
mcmc_sampler(n) = mcmc(p, rand(), n = n, stype = Float32, sig = 15);
smc_sampler(n) = reshape(smcsampler(p, N=n, p = d, σ = 15.0)[d, :], n);

# get mcmc samples
tic()
mcmc_stab, mcmc_errs = perform_experiment(mcmc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
mcmc_walltime = toc();
mcmc_time = [i * mcmc_walltime/n/n_experiments for i in 1:n]
size(mcmc_errs)
mcmc_er_mean = reshape(mean(mcmc_errs, 1), n, 2)
mcmc_er_sd = reshape(std(mcmc_errs, 1), n, 2)

# get smc samples
tic()
smc_stab, smc_errs = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime = toc();
size(smc_errs)
smc_time = [i * smc_walltime/n/n_experiments for i in 1:n]
smc_er_mean = reshape(mean(smc_errs, 1), n, 2)
smc_er_sd = reshape(std(smc_errs, 1), n, 2)


# Plot figures
fig_x = 9;
fig_y = 5;

println("Plotting Figures 1/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt = plt.plot(1:n, smc_er_mean[:,1], color = cols[1]);
mcmc_plt = plt.plot(1:n, mcmc_er_mean[:,1], color = cols[4]);
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
