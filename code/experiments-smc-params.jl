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

# get smc samples
println("Getting σ = 5")
smc_sampler(n) = reshape(smcsampler(p, N=n, p = 5, σ = 5.0)[d, :], n);
tic()
smc_stab, smc_errs_5 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_5 = toc();
smc_time_5 = [i * smc_walltime_5/n/n_experiments for i in 1:n]
smc_er_mean_5 = reshape(mean(smc_errs_5, 1), n, 2)
smc_er_sd_5 = reshape(std(smc_errs_5, 1), n, 2)

# get smc samples
println("Getting σ = 10")
smc_sampler(n) = reshape(smcsampler(p, N=n, p = 5, σ = 10.0)[d, :], n);
tic()
smc_stab, smc_errs_10 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_10 = toc();
smc_time_10 = [i * smc_walltime_10/n/n_experiments for i in 1:n]
smc_er_mean_10 = reshape(mean(smc_errs_10, 1), n, 2)
smc_er_sd_10 = reshape(std(smc_errs_10, 1), n, 2)

# get smc samples
println("Getting σ = 15")
smc_sampler(n) = reshape(smcsampler(p, N=n, p = 5, σ = 15.0)[d, :], n);
tic()
smc_stab, smc_errs_15 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_15 = toc();
smc_time_15 = [i * smc_walltime_15/n/n_experiments for i in 1:n]
smc_er_mean_15 = reshape(mean(smc_errs_15, 1), n, 2)
smc_er_sd_15 = reshape(std(smc_errs_15, 1), n, 2)

# get smc samples
println("Getting σ = 20")
smc_sampler(n) = reshape(smcsampler(p, N=n, p = 5, σ = 20.0)[d, :], n);
tic()
smc_stab, smc_errs_20 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_20 = toc();
smc_time_20 = [i * smc_walltime_20/n/n_experiments for i in 1:n]
smc_er_mean_20 = reshape(mean(smc_errs_20, 1), n, 2)
smc_er_sd_20 = reshape(std(smc_errs_20, 1), n, 2)

# get smc samples
println("Getting σ = 30")
smc_sampler(n) = reshape(smcsampler(p, N=n, p = 5, σ = 30.0)[d, :], n);
tic()
smc_stab, smc_errs_30 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_30 = toc();
smc_time_30 = [i * smc_walltime_30/n/n_experiments for i in 1:n]
smc_er_mean_30 = reshape(mean(smc_errs_30, 1), n, 2)
smc_er_sd_30 = reshape(std(smc_errs_30, 1), n, 2)

# Plot figures
fig_x = 9;
fig_y = 5;

println("Plotting Figures 1/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt_5 = plt.plot(smc_time_5, smc_er_mean_5[:,1], color = cols[1]);
smc_plt_10 = plt.plot(smc_time_10, smc_er_mean_10[:,1], color = cols[5]);
smc_plt_15 = plt.plot(smc_time_15, smc_er_mean_15[:,1], color = cols[15]);
smc_plt_20 = plt.plot(smc_time_20, smc_er_mean_20[:,1], color = cols[20]);
smc_plt_30 = plt.plot(smc_time_30, smc_er_mean_30[:,1], color = cols[4]);
#mcmc_plt = plt.plot(1:n, mcmc_er_mean[:,1], color = cols[4]);
plt.title(L"Mean absolute error of $E[x]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Walltime (s)", family = "serif");
plt.legend([smc_plt_5, smc_plt_10, smc_plt_15, smc_plt_20, smc_plt_30],[L"SMC Sampler - $\sigma = 5$",
                                                             L"SMC Sampler - $\sigma = 10$",
                                                             L"SMC Sampler - $\sigma = 15$",
                                                             L"SMC Sampler - $\sigma = 20$",
                                                             L"SMC Sampler - $\sigma = 30$"],
           frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/SMC_sig.png")
savefig("$path/SMC_sig.pdf")


# get smc samples
println("Getting d = 3")
d = 3;
smc_sampler(n) = reshape(smcsampler(p, N=n, p = d, σ = 15.0)[d, :], n);
tic()
smc_stab, smc_errs_d3 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_d3 = toc();
smc_time_d3 = [i * smc_walltime_d3/n/n_experiments for i in 1:n]
smc_er_mean_d3 = reshape(mean(smc_errs_d3, 1), n, 2)
smc_er_sd_d3 = reshape(std(smc_errs_d3, 1), n, 2)

# get smc samples
println("Getting d = 5")
d = 5;
smc_sampler(n) = reshape(smcsampler(p, N=n, p = d, σ = 15.0)[d, :], n);
tic()
smc_stab, smc_errs_d5 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_d5 = toc();
smc_time_d5 = [i * smc_walltime_d5/n/n_experiments for i in 1:n]
smc_er_mean_d5 = reshape(mean(smc_errs_d5, 1), n, 2)
smc_er_sd_d5 = reshape(std(smc_errs_d5, 1), n, 2)

# get smc samples
println("Getting d = 8")
d = 8;
smc_sampler(n) = reshape(smcsampler(p, N=n, p = d, σ = 15.0)[d, :], n);
tic()
smc_stab, smc_errs_d8 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_d8 = toc();
smc_time_d8 = [i * smc_walltime_d8/n/n_experiments for i in 1:n]
smc_er_mean_d8 = reshape(mean(smc_errs_d8, 1), n, 2)
smc_er_sd_d8 = reshape(std(smc_errs_d8, 1), n, 2)

# get smc samples
println("Getting d = 10")
d = 10;
smc_sampler(n) = reshape(smcsampler(p, N=n, p = d, σ = 15.0)[d, :], n);
tic()
smc_stab, smc_errs_d10 = perform_experiment(smc_sampler, n_experiments, n, function_list, true_values, col_amount = 1);
smc_walltime_d10 = toc();
smc_time_d10 = [i * smc_walltime_d10/n/n_experiments for i in 1:n]
smc_er_mean_d10 = reshape(mean(smc_errs_d10, 1), n, 2)
smc_er_sd_d10 = reshape(std(smc_errs_d10, 1), n, 2)

# Plot figures
fig_x = 9;
fig_y = 5;

println("Plotting Figures 1/8")
plt.figure(1, figsize=(fig_x,fig_y));
plt.clf();
plt.yscale("log");
smc_plt_d3 = plt.plot(smc_time_d3, smc_er_mean_d3[:,1], color = cols[7]);
smc_plt_d5 = plt.plot(smc_time_d5, smc_er_mean_d5[:,1], color = cols[6]);
smc_plt_d8 = plt.plot(smc_time_d8, smc_er_mean_d8[:,1], color = cols[13]);
smc_plt_d10 = plt.plot(smc_time_d10, smc_er_mean_d10[:,1], color = cols[19]);
#mcmc_plt = plt.plot(1:n, mcmc_er_mean[:,1], color = cols[4]);
plt.title(L"Mean absolute error of $E[x]$", family = "serif");
plt.ylabel("Mean error", family = "serif");
plt.xlabel("Walltime (s)", family = "serif");
plt.legend([smc_plt_d3, smc_plt_d5, smc_plt_d8, smc_plt_d10],[L"SMC Sampler - $p = 3$",
                                                             L"SMC Sampler - $p = 5$",
                                                             L"SMC Sampler - $p = 8$",
                                                             L"SMC Sampler - $p = 10$"],
           frameon = false, prop={"size"=>12, "family"=>"serif"})
savefig("$path/SMC_p.png")
savefig("$path/SMC_p.pdf")
