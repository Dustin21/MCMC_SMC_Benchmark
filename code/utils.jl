default_f(x) = x
function diff_from_function(x::Array, f=default_f)
  out = zeros(Float64, length(x));
  n = length(x);
  for i = 1:n
    out[i] = sum(f(x[1:i]))/i;
  end
  out
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

function means(x, functions = [])
  summaries = zeros(Float64, length(x), length(functions));
  for (i, fun) in enumerate(functions)
    summaries[:, i] = fast_means(x, fun);
  end
  return(summaries)
end


function fast_means(x::Array, f=default_f)
  out = zeros(Float64, length(x));
  n = length(x);
  out[1] = f(x[1]);
  for i = 2:n
    out[i] = (i-1)/i*out[i-1] + f(x[i, :])/i;
  end
  out
end

function collapsed_matrix(x, amount = 100)
  m,n = size(x);
  d = int(m/amount)
  collapse = zeros(Float64, d, n);
  for i = 1:d
    collapse[i,:] = mean(x[(i-1)*amount + 1: (i)*amount, :], 1)
  end
  return collapse
end

function stabalise(x::Array, target::Float64, range=0.01)
  n = length(x);
  first = 0;
  for i = n:-1:1
    if abs(x[i] - target)/target > range
      first = i;
      break;
    end
  end
  return(first)
end

## Colours for plots

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
cols = Tuple[]
for i = 1:20
  push!(cols, (tableau20[i][1]/255.0, tableau20[i][2]/255.0, tableau20[i][3]/255.0));
end
