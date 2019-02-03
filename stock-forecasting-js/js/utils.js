function dropout_nn(x,keep_prob){
  uniform = tf.randomUniform(x.shape)
  added = tf.add(tf.scalar(keep_prob),uniform)
  binary = tf.floor(added)
  return tf.mul(tf.div(x,tf.scalar(keep_prob)),binary)
}

function dropout_lstm(cell,a,h,c,dropout_input=1.0,dropout_output=1.0){
  var outputs = []
  for(var i = 0; i < a.shape[1];i++){
    var start = a.slice([0,i,0],[-1,1,-1]).reshape([-1,a.shape[2]])
    if(dropout_input< 1) start = dropout_nn(start,dropout_input)
    applied=cell.apply([start,h,c])
    if(dropout_output<1) applied[0] = dropout_nn(applied[0],dropout_output)
    outputs.push(applied[0].reshape([-1,1,applied[1].shape[1]]))
    h = applied[1]
    c = applied[2]
  }
  return [tf.concat(outputs,1),h,c]
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
async function async_sleep(ms) {
  await sleep(ms);
}
function wasting_time(count){
  for(var n = 0; n < count; n++){
    // do nothing
  }
}
function arange(start, end, skip){
  var arr = [start]
  while((parseFloat(arr.slice(-1))+parseFloat(skip)) < (end)) arr.push(parseFloat(arr.slice(-1))+parseFloat(skip))
  return arr
}
async function tf_tolist(a){
  var arr = []
  for(var i = 0; i < a.shape[0];i++)arr.push(Array.prototype.slice.call(await a.slice([0,0],[1,-1]).data()));
  return arr
}
function tf_str_tolist(a){
  return JSON.parse(a.toString().slice(7).trim())
}
function tf_slice_tolist(a){
  var arr = []
  for(var i = 0; i < a.shape[0];i++) {
    var val = JSON.parse(a.slice([i,0],[1,1]).toString().slice(7).trim().replace(',',''))[0][0]
    arr.push(val);
  }
  return arr
}
function tf_nj_list(a){
  var arr = nj.zeros([a.shape[0],a.shape[1]]).tolist();
  for(var i = 0; i < a.shape[0];i++){
    for(var k = 0; k < a.shape[1];k++) arr[i][k] = JSON.parse(a.slice([i,k],[1,1]).toString().slice(7).trim().replace(',',''))[0][0]
  }
  return arr
}
function tf_nj_list_flatten(a){
  var arr = nj.zeros([a.shape[0]]).tolist();
  for(var i = 0; i < a.shape[0];i++) arr[i] = JSON.parse(a.slice([i],[1]).toString().slice(7).trim())[0]
  return arr
}
function label_encoder(arr){
  var unique = [...new Set(arr)];
  var encoder = []
  for(var i = 0; i < arr.length;i++) encoder.push(unique.indexOf(arr[i]))
  return {'unique':unique,'encode':encoder}
}
function shuffle(arr1, arr2) {
  var index = arr1.length;
  var rnd, tmp1, tmp2;
  while (index) {
    rnd = Math.floor(Math.random() * index);
    index -= 1;
    tmp1 = arr1[index];
    tmp2 = arr2[index];
    arr1[index] = arr1[rnd];
    arr2[index] = arr2[rnd];
    arr1[rnd] = tmp1;
    arr2[rnd] = tmp2;
  }
}
function get_index(arr, val){
  var indices = []
  for(var i = 0; i < arr.length;i++) if(arr[i] == val) indices.push(i)
  return indices
}
function get_elements(arr, indices){
  var elements = []
  for (i in indices) {
    elements.push(arr[indices[i]])
  }
  return elements
}
function pca(a, n_components){
  a = tf.tensor(a)
  tiled=tf.matMul(tf.ones([150,1]), a.mean(0).reshape([1,-1]))
  sub = tf.sub(a,tiled)
  sub_list = tf_str_tolist(tf.matMul(sub.transpose(),sub))
  eig=numeric.eig(sub_list)
  eigenvectors = tf.tensor(eig.E.x).slice([0,0],[-1,n_components])
  return tf.matMul(sub, eigenvectors)
}
function svd(a, n_components){
  output_svd = numeric.svd(a)
  tensor_U = tf.tensor(output_svd['U'])
  tensor_V = tf.tensor(output_svd['V']).slice([0,0],[-1,n_components])
  return tf.matMul(tensor_U, tensor_V)
}
function nnmf(arr, n_components){
  a = tf.tensor(arr)
  var var_H = tf.randomNormal([n_components,a.shape[1]])
  var var_W = tf.randomNormal([a.shape[0],n_components])
  var_H = tf.variable(var_H,trainable=true)
  var_W = tf.variable(var_W,trainable=true)
  var f = () => var_W.matMul(var_H);
  var cost = (pred, label) => tf.square(tf.sub(label,pred)).mean();
  var optimizer = tf.train.adam(1);
  for (var i = 0; i < 100; i++) {
    cost(f(), a).print()
    optimizer.minimize(() => cost(f(), a));
  }
  return tf_nj_list(var_W)
}
function metrics(a){
  a = tf.tensor(a)
  squared = tf.square(tf.sub(a,a.mean(0))).sum(0)
  variance = tf.div(squared,tf.scalar(a.shape[0]-1))
  std = tf.sqrt(tf.div(squared,tf.scalar(a.shape[0]-1)))
  return {'std':std,'variance':variance}
}
function standard_scaling(a){
  squared = tf.square(tf.sub(a,a.mean(0))).sum(0)
  variance = tf.div(squared,tf.scalar(a.shape[0]-1))
  std = tf.sqrt(tf.div(squared,tf.scalar(a.shape[0]-1)))
  return tf.div(tf.sub(x,x.mean(0)),std)
}
function minmax_scaling(a){
  a = tf.tensor(a)
  return tf.div(tf.sub(a,a.min(0)), tf.sub(a.max(0),a.min(0)))
}
function minmax_1d(a){
  a = tf.tensor(a)
  a_min = tf_str_tolist(a.min())
  a_max = tf_str_tolist(a.max())
  scaled = tf.div(tf.sub(a,a.min()), tf.sub(a.max(),a.min()))
  return {'scaled':scaled,'min':a_min,'max':a_max}
}
function reverse_minmax_1d(a, a_min, a_max){
  return tf.add(tf.mul(a, tf.scalar(a_max-a_min)), tf.scalar(a_min))
}
function one_hot(label_encoder){
  var onehot = nj.zeros([label_encoder['encode'].length,label_encoder['unique'].length]).tolist();
  for(var i = 0; i < label_encoder['encode'].length;i++) onehot[i][label_encoder['encode'][i]] = 1
  return onehot
}
function plot_map(data, X_mean, Y_mean, arr_X, arr_Y){
  var data_map = [
    {
      z: data['z'],
      x: data['xx'],
      y: data['y_'],
      type: 'heatmap',
      opacity: 0.4,
      colorscale: 'Jet',
      colorbar: {
        title: 'Label',
        titleside: 'top',
        tickvals: [...Array(data['label'].length).keys()],
        ticktext: data['label']
      }
    },
    {
      x: data['x'],
      y: data['y'],
      mode: 'markers',
      type: 'scatter',
      marker: {
        colorscale: 'Jet',
        color: data['color']
      }
    }
  ];
  var layout = {
    title: 'Decision Boundaries',
    showlegend: false,
    annotations: []
  }
  for(var i =0; i <data['label'].length;i++){
    data_map.push({
      x: [X_mean, arr_X[i]],
      y: [Y_mean, arr_Y[i]],
      mode: 'lines',
      line: {
        color: 'rgb(0,0,0)',
        width: 1
      }
    })
    layout['annotations'].push({
      x: arr_X[i],
      y: arr_Y[i],
      xref: 'x',
      yref: 'y',
      text: data['label'][i],
      showarrow: true,
      arrowhead: 3,
      ax: 0,
      ay: -20,
      arrowside:'start',
      font: {
        size: 16
      },
    })
  }
  Plotly.newPlot('div_output', data_map, layout);
}

function plot_graph(data,with_acc=true){
  var trace_loss = {
    x: data['epoch'],
    y: data['loss'],
    mode: 'lines',
    type: 'scatter'
  }
  var layout_loss = {
    'title': 'Loss Graph',
    xaxis: {
      autotick: true
    },
    margin: {
      b: 25,
      pad: 4,
      l:25
    }
  }
  var trace_acc = {
    x: data['epoch'],
    y: data['accuracy'],
    mode: 'lines',
    type: 'scatter',
    name: 'Training accuracy'
  }
  var layout_acc = {
    'title': 'Accuracy Graph',
    xaxis: {
      autotick: true
    },
    margin: {
      b: 25,
      pad: 4,
      l:25
    }
  }
  Plotly.newPlot('div_loss', [trace_loss], layout_loss);
  if(with_acc)Plotly.newPlot('div_acc', [trace_acc], layout_acc);
}
function plot_joyplot(x_outside,div,title,btm_gap=0.1, top_gap=0.25, gap=0.1,ratio=1.0){
  concat_x = [], concat_y = []
  for (var out = 0; out < x_outside.length; out++) {
    num_bins = Math.ceil(Math.sqrt(x_outside[out].length));
    bins = d3.layout.histogram().frequency(false).bins(num_bins)(x_outside[out])
    new_x = [], new_y = []
    for(var i = 0; i < bins.length;i++){
      new_x.push((bins[i]['dx']/2)+bins[i]['x'])
      new_y.push(bins[i]['y'])
    }

    if(out == 0){
      for (var i = 0;i < new_y.length;i++) {
        new_y[i] += 0.1;
      }
    }
    else{
      for (var i = 0;i < new_y.length;i++) {
        new_y[i] += (0.1+0.1*out);
      }
    }
    concat_y.push(new_y)
    concat_x.push(new_x)
  }
  var data_joyplot = [], out_min_x = 0, out_max_x = 0
  for (var out = 0; out < x_outside.length; out++) {

    min_y = Math.min(...concat_y[out]), max_y = Math.max(...concat_y[out])
    min_x = Math.min(...concat_x[out]), max_x = Math.max(...concat_x[out])
    if(min_x > out_min_x) out_min_x = min_x
    if(max_x > out_max_x) out_max_x = max_x
    data_joyplot.push({
      y:[min_y,min_y],
      x:[min_x,max_x],
      line:{
        color: '#FFFFFF',
        width: 0.1
      },
      type:'scatter',
      mode:'lines'
    })
    mul_concat_y = []
    for (var k = 0; k < concat_y[out].length; k++) mul_concat_y[k] = concat_y[out][k] * ratio
    data_joyplot.push({
      name:out,
      fillcolor:'rgba(222, 34, 36, 0.8)',
      mode: 'lines',
      y:mul_concat_y,
      x:concat_x[out],
      line:{
        color: '#FFFFFF',
        width: 0.5,
        shape: 'spline'
      },
      type:'scatter',
      fill:'tonexty'
    })
  }
  tickvals = []
  for(var i = 0; i < concat_y.length;i++) tickvals.push(Math.min(...concat_y[i]))
  var layout={
    "title":title,
    "yaxis":{
      "title":"epoch",
      "ticklen":4,
      "gridwidth":1,
      "showgrid":true,
      "range":[
        0,
        Math.min(...concat_y[concat_y.length-1]) +0.25
      ],
      "gridcolor":"rgb(255,255,255)",
      "zeroline":false,
      "showline":false,
      "ticktext":arange(0,concat_y.length,1),
      "tickvals":tickvals
    },
    "showlegend":false,
    "xaxis":{
      "title":"tensor values",
      "ticklen":4,
      "dtick":0.1,
      "showgrid":false,
      "range":[out_min_x, out_max_x + 0.05],
      "zeroline":false,
      "showline":false
    },
    "hovermode":"closest",
    "font":{
      "family":"Balto"
    },
    margin: {
      b: 50,
      t: 25,
      pad: 4,
      l:50
    }
  }
  Plotly.newPlot(div, data_joyplot, layout);
}
function kernelDensityEstimator(kernel, x) {
  return function(sample) {
    return x.map(function(x) {
		return [x, d3.mean(sample, function(v) { return kernel(x - v); })];
    });
  };
}
function epanechnikovKernel(bandwith) {
  return function(u) {
	if(Math.abs(u = u /  bandwith) <= 1) {
	 return 0.75 * (1 - u * u) / bandwith;
	} else return 0;
  };
}
function histogram(arr,bins=30,norm=true,density=false,jitter=0.001){
  var max_arr = Math.max(...arr)
  var min_arr = Math.min(...arr)
  var arr_bins = []
  var start = min_arr
  var skip = (max_arr-min_arr)/bins
  var x_arange = []
  while(arr_bins.length<bins){
    arr_bins.push([start+jitter,start+skip])
    x_arange.push((start+jitter+start+skip)/2)
    start += skip
  }
  var hist = nj.zeros([bins]).tolist()
  for(var i = 0; i < arr.length;i++){
    for(var b = 0; b < arr_bins.length;b++){
      if(arr[i] >= arr_bins[b][0] && arr[i] <= arr_bins[b][1]){
        hist[b] += 1
        break
      }
    }
  }
  function getSum(total, num) {
    return total + num;
  }
  //sum_hist = hist.reduce(getSum)
  if(norm) for(var b = 0; b < arr_bins.length;b++) hist[b] /= arr.length
  if(density) for(var b = 0; b < arr_bins.length;b++) hist[b] /= (arr.length*bins)
  return {'y':hist,'x':x_arange}
}
function plot_regression(data){
  var data_map = [
    {
      x: data['x'],
      y: data['y'],
      name: data['name'],
      mode: 'markers',
      type: 'scatter',
      marker: {
        color: 'red'
      }
    },
    {
      x: data['x-line'],
      y: data['y-line'],
      mode: 'lines',
      name: 'linear regressed',
      type: 'scatter',
      line: {
        color: 'blue',
      }
    }
  ];
  var layout = {
    title: data['title'],
    showlegend: true,
    xaxis:{
      title:data['x-title']
    },
    yaxis:{
      title:data['y-title']
    }
  }
  Plotly.newPlot(data['div'], data_map, layout);
}
function plot_compare_distribution(data_arr, labels, colors, div){
  data_plot = []
  for(var outer = 0; outer < data_arr.length;outer++){
    data = data_arr[outer]
    data_y = []
    for(var i = 0; i < data.length;i++)data_y.push(labels[outer])
    max_arr = Math.max(...data)
    min_arr = Math.min(...data)
    num_bins = Math.ceil(Math.sqrt(data.length));
    kde = kernelDensityEstimator(epanechnikovKernel(max_arr/50), arange(min_arr,max_arr,(max_arr-min_arr)/num_bins))
    kde = kde(data)
    bar_x = [], bar_y = []
    for(var i = 0; i < kde.length;i++){
      bar_x.push(kde[i][0])
      bar_y.push(kde[i][1])
    }
    min_line_y = Math.min(...bar_y)
    for(var i = 0; i < bar_y.length;i++) bar_y[i] -= min_line_y
    kde = kernelDensityEstimator(epanechnikovKernel(max_arr/7), arange(min_arr,max_arr,(max_arr-min_arr)/128))
    kde = kde(data)
    line_x = [], line_y = []
    for(var i = 0; i < kde.length;i++){
      line_x.push(kde[i][0])
      line_y.push(kde[i][1])
    }
    min_line_y = Math.min(...line_y)
    for(var i = 0; i < line_y.length;i++) line_y[i] -= min_line_y
    data_plot.push({
      opacity:0.7,
      legendgroup:labels[outer],
      autobinx:false,
      name:labels[outer],
      yaxis:'y1',
      xaxis:'x1',
      marker:{
        color:colors[outer]
      },
      type:'bar',
      x:bar_x,
      y:bar_y
    })
    data_plot.push({
      showlegend:false,
      legendgroup:labels[outer],
      name: labels[outer],
      yaxis:'y1',
      xaxis:'x1',
      marker:{
        color:colors[outer]
      },
      mode:'lines',
      type:'scatter',
      x:line_x,
      y:line_y
    })
    data_plot.push({
      showlegend:false,
      legendgroup:labels[outer],
      name:labels[outer],
      yaxis:'y2',
      xaxis:'x1',
      marker:{
        color:colors[outer],
        symbol:'line-ns-open'
      },
      mode:'markers',
      x:data,
      y:data_y,
      type:'scatter',
      text:null
    })
  }
  layout_plot={"yaxis1": {"position": 0.0, "domain": [0.1, 1], "anchor": "free"}, "title": "Distribution plot", "xaxis1": {"zeroline": false, "domain": [0.0, 1.0], "anchor": "y2"},
  "barmode": "overlay", "yaxis2": {"domain": [0, 0.10], "showticklabels": false, "anchor": "x1", "dtick": 1}, "hovermode": "closest", "legend": {"traceorder": "reversed"}}
  Plotly.newPlot(div, data_plot,layout_plot);
}
function simple_investor(real_signal,predicted_signal,delay,initial_money,max_buy,max_sell,dates){
  outputs = []
  current_decision = 0
  current_val = predicted_signal[0]
  states_sell_X = []
  states_buy_X = []
  states_buy_index = []
  states_sell_Y = []
  states_buy_Y = []
  current_inventory = 0
  state=1
  starting_money = initial_money
  function buy(i,initial_money,current_inventory){
    if(i < real_signal.length) shares = Math.floor(initial_money / real_signal[i]);
    else shares = Math.floor(initial_money / predicted_signal[i])
    if(shares < 1){} //outputs.push('day '+i+': total balances '+initial_money+', not enough money to buy a unit price '+real_signal[i])
    else{
      if(shares>max_buy)buy_units=max_buy
      else buy_units=shares
      if(i < real_signal.length) gains = buy_units*real_signal[i]
      else gains = buy_units*predicted_signal[i]
      initial_money -= gains
      current_inventory += buy_units
      outputs.push("<tr><td>"+dates[i]+"</td><td>buy "+buy_units+" units</td><td>"+gains+"</td><td>NULL</td><td>"+initial_money+"</td></tr>")
      states_buy_X.push(dates[i])
      states_buy_index.push(i)
      if(i < real_signal.length) states_buy_Y.push(real_signal[i]);
      else states_buy_Y.push(predicted_signal[i]);
    }
    return [initial_money,current_inventory]
  }
  if(state==1){
    bought = buy(0, initial_money, current_inventory)
    initial_money = bought[0]
    current_inventory = bought[1]
  }
  for(var i = 1;i<predicted_signal.length;i++){
    if(predicted_signal[i] < current_val && state == 0 && (predicted_signal.length-i) > delay){
      if(current_decision < delay) current_decision++;
      else{
        state = 1
        bought = buy(i, initial_money, current_inventory)
        initial_money = bought[0]
        current_inventory = bought[1]
        current_decision = 0
      }
    }
    if((predicted_signal[i] > current_val && state == 1)||((predicted_signal.length-i) < delay && state == 1)){
      if(current_decision < delay) current_decision++;
      else{
        state = 0
        if(current_inventory == 0){}//outputs.push(dates[i]+': cannot sell anything, inventory 0')
        else{
          if(current_inventory > max_sell)sell_units = max_sell;
          else sell_units = current_inventory;
          current_inventory -= sell_units
          if(i < real_signal.length) total_sell = sell_units * real_signal[i]
          else total_sell = sell_units * predicted_signal[i]
          initial_money += total_sell
          try {
            if(i < real_signal.length) invest = ((real_signal[i] - real_signal[states_buy_index[states_buy_index.length-1]]) / real_signal[states_buy_index[states_buy_index.length-1]]) * 100
            else invest = ((predicted_signal[i] - predicted_signal[states_buy_index[states_buy_index.length-1]]) / predicted_signal[states_buy_index[states_buy_index.length-1]]) * 100
          }
          catch(err) {invest = 0}
          outputs.push("<tr><td>"+dates[i]+"</td><td>sell "+sell_units+" units</td><td>"+total_sell+"</td><td>"+invest+"%</td><td>"+initial_money+"</td></tr>")
        }
        current_decision = 0
        states_sell_X.push(dates[i])
        if(i < real_signal.length) states_sell_Y.push(real_signal[i])
        else states_sell_Y.push(predicted_signal[i])
      }
    }
    current_val = predicted_signal[i]
  }
  invest = ((initial_money - starting_money) / starting_money) * 100
  return {'overall gain':(initial_money-starting_money),'overall investment':invest,
  'sell_Y':states_sell_Y,'sell_X':states_sell_X,'buy_Y':states_buy_Y,'buy_X':states_buy_X,'output':outputs}
}
