using Distributions: Normal, Uniform, pdf
using PyPlot
include("mcmc.jl")
include("utils.jl")
include("SMCSampler.jl")
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
cols = Tuple[]
for i = 1:20
  push!(cols, (tableau20[i][1]/255.0, tableau20[i][2]/255.0, tableau20[i][3]/255.0));
end

p(x, alpha = 1.0) = (exp(-(x - 2).^2/2) +
                       0.5*exp(-(x+2).^2/1) +
                       0.5*exp(-(x-5).^2/1) +
                       0.5*exp(-(x-15).^2/1)).^(alpha).*exp(-(x).^2/2).^(1-alpha);
e_x = 4.05887;
e_x2 = 46.2633;

srand(1236)
path = "/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/520\ -\ All\ about\ that\ bayes/BenchmarkingProject/Report/plots"
x0 = rand(); #starting position
n = int(1e6); #number of samples

##  MCMC
@time x = mcmc(p, x0, n = n, stype = Float32, sig = 15);
@time ys = smcsampler(p, N=n, p = 5, σ = 15.0);
y = reshape(ys[5,:], n);
fig_x = 9;
fig_y = 5;
plt.figure(1, figsize=(fig_x,fig_y));
plt.rc("font", family="serif")
plt.hist(x,100, alpha=0.4, normed=true, color = cols[7], edgecolor = "none");
plt.title("MCMC output with $n samples", family = "serif");
plt.ylabel("Frequency", family = "serif");
plt.xlabel(L"$x$", family = "serif");
savefig("$path/Hist.png")

g(x) = x.^2;
diff_x = fast_means(x);
diff_x2 = fast_means(x, g);
stable_x = stabalise(diff_x, e_x);

diff_y = fast_means(y);
diff_y2 = fast_means(y, g);
stable_y = stabalise(diff_y, e_x);

plt.clf()
plt.plot((1, n), (e_x, e_x), color = cols[19])
plt.plot((1, n), (e_x*1.01, e_x*1.01), color = cols[20])
plt.plot((1, n), (e_x*0.99, e_x*0.99), color = cols[20])
plt.plot((stable_x, stable_x), (0, e_x*20), color = cols[7])
plt.plot((stable_y, stable_y), (0, e_x*20), color = cols[7])
mcmc_plt = plt.plot(1:n, diff_x, color = cols[1])
smc_plt = plt.plot(1:n, diff_y, color = cols[3])
plt.ylim(e_x*0.98, e_x*1.02)
plt.legend([smc_plt, mcmc_plt], ["SMC Sampler", "MCMC"], frameon = false, prop={"size"=>12, "family"=>"serif"})
plt.xlabel("Number iterations", family = "serif");
plt.xlim(0, n)
plt.title(L"Convergence a typical run to $E[x]$")
savefig("$path/Convergence.png")


x1 = linspace(-6, 20, 500)
plt.clf()
plt.figure(2, figsize = (fig_x, fig_y))
plt.plot(x1, p(x1), color = cols[1])
plt.xlim(-6, 20)
plt.ylim(0,1.05)
savefig("$path/function.pdf")