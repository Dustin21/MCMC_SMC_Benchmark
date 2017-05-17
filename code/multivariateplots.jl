using PyPlot
means = [-2.0, 2.0, -5.0, -15.0]
sds = [1.0, 0.5, 0.5, 0.5]
weights = [1.0, 0.5, 0.5, 0.5]
function mix_gaussians(x, means = means, sds = sds, weights = weights, alpha = 1.0)
    l = length(means);
    out = 0;
    for i = 1:l
        out = out + weights[i] * exp(-(x + means[i]).^2 / (2*sds[i]))
    end
    return out.^(alpha).*exp(-(x).^2/2).^(1-alpha)
end

function twod(x1, x2 = 1.0, alpha = 1.0)
    m1 = [-2.0, 2.0, -5.0, -8.0]
    sd1 = [1.0, 0.5, 0.5, 0.5]
    w1 = [1.0, 0.5, 0.3, 0.5]
    m2 = [2.0, -8.0, -5.0, 7.0]
    sd2 = [0.5, 1.5, 1.2, 0.3]
    w2 = [2.0, 0.5, 1.0, 0.5]
    return (mix_gaussians(x1, m1, sd1, w1) .* mix_gaussians(x2, m2, sd2, w2)).^alpha .* (mix_gaussians(x1, m1, sd1, w1, 0.0)*mix_gaussians(x2, m1, sd1, w1, 0.0)).^(1-alpha)
end

function surfplot(fx, x1lim, x2lim; alpha = 1.0, n = 400, figsize = [10, 10], filename = "3dplot.png", path = "/Users/jasonhartford/MediaFire/Documents/ComputerScience/UBC/520\ -\ All\ about\ that\ bayes/BenchmarkingProject/Report/plots")
  x1 = linspace(x1lim[1], x1lim[2], n)
  x2 = linspace(x2lim[1], x2lim[2],n)
  x1grid = repmat(x1',n,1)
  x2grid = repmat(x2,1,n)

  z = zeros(n,n)

  for i in 1:n
      for j in 1:n
          z[i,j] = fx(x1[i],x2[j], alpha)
      end
  end
  z = z./sum(z)
  plt.clf()
  fig = plt.figure("pyplot_surfaceplot",figsize=(figsize[1],figsize[2]))
  ax = fig[:add_subplot](2,1,1, projection = "3d")
  ax[:plot_surface](x1grid, x2grid, z, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("coolwarm"), alpha=0.8, linewidth=0.05)
  xlabel(L"$X_1$")
  ylabel(L"$X_2$")
  zlim(0,0.0004)
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



surfplot(twod, [-10, 10], [-10, 10], n = 400, figsize = [10, 5])

for i = 0.0:0.01:1.0
  println("Working on $i")
  surfplot(twod, [-10, 10], [-10, 10], n = 400, figsize = [10, 5], alpha = i, filename =  @sprintf("Normalised-Mix-Gaussian-anealing-animation%0.2f.png", i))
end

st = @sprintf("%0.2f", 3.1)
println(@sprintf("working on $st"))

n = 400
x1 = linspace(-10, 10, n)
x2 = linspace(-10, 10,n)
x1grid = repmat(x1',n,1)
x2grid = repmat(x2,1,n)

z = zeros(n,n)

for i in 1:n
  for j in 1:n
    z[i,j] = twod(x1[i],x2[j])
  end
end
figsize = [10, 10]
fig = figure("pyplot_surfaceplot",figsize=(figsize[1],figsize[2]))
ax = fig[:add_subplot](2,1,1, projection = "3d")
ax[:plot_surface](x1grid, x2grid, z, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("coolwarm"), alpha=0.8, linewidth=0.15)
xlabel(L"$X_1$")
ylabel(L"$X_2$")
title("Multivariate Product Space")

subplot(212)
ax = fig[:add_subplot](2,1,2)
cp = ax[:contour](x1grid, x2grid, z, cmap=ColorMap("coolwarm"), linewidth=1.0)
ax[:clabel](cp, inline=1, fontsize=10)
xlabel(L"$X_1$")
ylabel(L"$X_2$")
title("Contour Plot")
tight_layout()