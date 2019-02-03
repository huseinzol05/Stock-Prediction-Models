var color_list = ['#c23531','#2f4554', '#61a0a8', '#d48265', '#91c7ae','#749f83',  '#ca8622', '#bda29a','#6e7074', '#546570', '#c4ccd3'];
var colors = ['#5793f3', '#d14a61', '#675bba','#b62f46'];
var close = GOOGLE['data'].map(function(el, idx) {
  return el[1];
})
var stocks = GOOGLE['data'].map(function(el, idx) {
  return [el[0],el[1],el[3],el[2]];
})
var stock_date = GOOGLE['date'];
var volume = GOOGLE['volume'];
var csv;
var indeces = {};
var dataMA5, dataMA10, dataMA20, dataMA30;
var total_investment, total_gain, stock_changes, stock_changes_percent

function smoothing_line(scalars,weight){
  last = scalars[0]
  smoothed = []
  for(var i = 0; i < scalars.length;i++){
    smoothed_val = last * weight + (1 - weight) * scalars[i]
    smoothed.push(smoothed_val)
    last = smoothed_val
  }
  return smoothed
}

function generate_investment(strings,values){
  colors = "";
  for(var i = 0; i < strings.length;i++){
    if(values[i]>=0) colors += "<div class='col s12 m2'><div class='card'><div class='card-content'><a class='btn-floating waves-effect waves-light green' style='width:100px;height:100px;margin-bottom:20px'><i class='material-icons' style='font-size:3rem; line-height:95px'>arrow_upward</i></a><p><h6>"+strings[i]+values[i]+"</h6></p></div></div></div>";
    else colors += "<div class='col s12 m2'><div class='card'><div class='card-content'><a class='btn-floating waves-effect waves-light red' style='width:100px;height:100px;margin-bottom:20px'><i class='material-icons' style='font-size:3rem; line-height:95px'>arrow_downward</i></a><p><h6>"+strings[i]+values[i]+"</h6></p></div></div></div>";
  }
  $('#color-investment').html(colors);
}

function buildConfig() {
  return {
    delimiter: $('#delimiter').val(),
    header: $('#header').prop('checked'),
    dynamicTyping: $('#dynamicTyping').prop('checked'),
    skipEmptyLines: $('#skipEmptyLines').prop('checked'),
    preview: parseInt($('#preview').val() || 0),
    step: $('#stream').prop('checked') ? stepFn : undefined,
    encoding: $('#encoding').val(),
    worker: $('#worker').prop('checked'),
    comments: $('#comments').val(),
    complete: completeFn,
    error: errorFn
  }
}

function errorFn(err, file) {
    Materialize.toast("ERROR: " + err + file,3000)
}

function completeFn(results) {
  if (results && results.errors) {
    if (results.errors) {
      errorCount = results.errors.length;
      firstError = results.errors[0]
    }
    if (results.data && results.data.length > 0)
    rowCount = results.data.length
  }
  csv = results['data'];
  for(var i = 0;i<csv[0].length;i++) indeces[csv[0][i].toLowerCase()] = i;
  stocks = [];
  volume = [];
  stock_date = [];
  for(var i = 1;i<csv.length;i++){
    if(!isNaN(csv[i][indeces['open']]) && !isNaN(csv[i][indeces['close']]) && !isNaN(csv[i][indeces['low']]) && !isNaN(csv[i][indeces['high']]) && !isNaN(csv[i][indeces['volume']])){
      stocks.push([parseFloat(csv[i][indeces['open']]),
      parseFloat(csv[i][indeces['close']]),
      parseFloat(csv[i][indeces['low']]),
      parseFloat(csv[i][indeces['high']])]);
      volume.push(csv[i][indeces['volume']]);
      stock_date.push(csv[i][indeces['date']]);
    }
  }
  close = stocks.map(function(el, idx) {
    return el[1];
  })
  plot_stock();
}

var csv, config = buildConfig();
$('#uploadcsv').change(function() {
    csv = null;
    file = document.getElementById('uploadcsv');
    if ($(this).val().search('.csv') <= 0) {
        $(this).val('');
        Materialize.toast('Only support CSV', 4000);
        return
    }
    $(this).parse({
        config: config
    })
})

function calculate_distribution(real,predict){
  data_plot = []
  data_arr = [real,predict]
  for(var outer = 0; outer < data_arr.length;outer++){
    data = data_arr[outer]
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
    data_plot.push({'bar_x':bar_x,'bar_y':bar_y})
  }
  option = {
    color: colors,

    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross'
      }
    },
    legend: {
      data:['real histogram','predict histogram']
    },
    xAxis: [
      {
        type: 'category',
        data: data_plot[0]['bar_x']
      },
      {
        type: 'category',
        data: data_plot[1]['bar_x']
      }
    ],
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name:'real histogram',
        type:'bar',
        data:data_plot[0]['bar_y']
      },
      {
        name:'predict histogram',
        type:'bar',
        data:data_plot[1]['bar_y'].slice(0,data_plot[1]['bar_y'].length-2)
      }
    ]
  };
  var bar_plot = echarts.init(document.getElementById('div_dist'));
  bar_plot.setOption(option,true);
}

function calculateMA(dayCount, data) {
  var result = [];
  for (var i = 0, len = data.length; i < len; i++) {
    if (i < dayCount) {
      result.push('-');
      continue;
    }
    var sum = 0;
    for (var j = 0; j < dayCount; j++) {
      sum += data[i - j][1];
    }
    result.push((sum / dayCount).toFixed(2));
  }
  return result;
}

function plot_stock(){
  dataMA5 = calculateMA(5, stocks);
  dataMA10 = calculateMA(10, stocks);
  dataMA20 = calculateMA(20, stocks);
  dataMA30 = calculateMA(30, stocks);
  option = {
    animation: false,
    color: color_list,
    title: {
      left: 'center'
    },
    legend: {
      top: 30,
      data: ['STOCK', 'MA5', 'MA10', 'MA20', 'MA30']
    },
    tooltip: {
      trigger: 'axis',
      position: function (pt) {
        return [pt[0], '10%'];
      }
    },
    axisPointer: {
      link: [{
        xAxisIndex: [0, 1]
      }]
    },
    dataZoom: [{
      type: 'slider',
      xAxisIndex: [0, 1],
      realtime: false,
      start: 0,
      end: 100,
      top: 65,
      height: 20,
      handleIcon: 'M10.7,11.9H9.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4h1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
      handleSize: '120%'
    }, {
      type: 'inside',
      xAxisIndex: [0, 1],
      start: 40,
      end: 70,
      top: 30,
      height: 20
    }],
    xAxis: [{
      type: 'category',
      data: stock_date,
      boundaryGap : false,
      axisLine: { lineStyle: { color: '#777' } },
      axisLabel: {
        formatter: function (value) {
          return echarts.format.formatTime('MM-dd', value);
        }
      },
      min: 'dataMin',
      max: 'dataMax',
      axisPointer: {
        show: true
      }
    }, {
      type: 'category',
      gridIndex: 1,
      data: stock_date,
      scale: true,
      boundaryGap : false,
      splitLine: {show: false},
      axisLabel: {show: false},
      axisTick: {show: false},
      axisLine: { lineStyle: { color: '#777' } },
      splitNumber: 20,
      min: 'dataMin',
      max: 'dataMax',
      axisPointer: {
        type: 'shadow',
        label: {show: false},
        triggerTooltip: true,
        handle: {
          show: true,
          margin: 30,
          color: '#B80C00'
        }
      }
    }],
    yAxis: [{
      scale: true,
      splitNumber: 2,
      axisLine: { lineStyle: { color: '#777' } },
      splitLine: { show: true },
      axisTick: { show: false },
      axisLabel: {
        inside: true,
        formatter: '{value}\n'
      }
    }, {
      scale: true,
      gridIndex: 1,
      splitNumber: 2,
      axisLabel: {show: false},
      axisLine: {show: false},
      axisTick: {show: false},
      splitLine: {show: false}
    }],
    grid: [{
      left: 20,
      right: 30,
      top: 110,
    }, {
      left: 20,
      right: 30,
      top: 400
    }],
    graphic: [{
      type: 'group',
      left: 'center',
      top: 70,
      width: 300,
      bounding: 'raw',
      children: [{
        id: 'MA5',
        type: 'text',
        style: {fill: color_list[1]},
        left: 0
      }, {
        id: 'MA10',
        type: 'text',
        style: {fill: color_list[2]},
        left: 'center'
      }, {
        id: 'MA20',
        type: 'text',
        style: {fill: color_list[3]},
        right: 0
      }]
    }],
    series: [{
      name: 'Volume',
      type: 'bar',
      xAxisIndex: 1,
      yAxisIndex: 1,
      itemStyle: {
        normal: {
          color: '#7fbe9e'
        },
        emphasis: {
          color: '#140'
        }
      },
      data: volume
    }, {
      type: 'candlestick',
      name: 'STOCK',
      data: stocks,
      itemStyle: {
        normal: {
          color: '#ef232a',
          color0: '#14b143',
          borderColor: '#ef232a',
          borderColor0: '#14b143'
        },
        emphasis: {
          color: 'black',
          color0: '#444',
          borderColor: 'black',
          borderColor0: '#444'
        }
      }
    }, {
      name: 'MA5',
      type: 'line',
      data: dataMA5,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        normal: {
          width: 1
        }
      }
    }, {
      name: 'MA10',
      type: 'line',
      data: dataMA10,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        normal: {
          width: 1
        }
      }
    }, {
      name: 'MA20',
      type: 'line',
      data: dataMA20,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        normal: {
          width: 1
        }
      }
    },
    {
      name: 'MA30',
      type: 'line',
      data: dataMA30,
      smooth: true,
      showSymbol: false,
      lineStyle: {
        normal: {
          width: 1
        }
      }
    }]
  };

  var chart_stock = echarts.init(document.getElementById('div_output'));
  chart_stock.setOption(option,true);
}
plot_stock();

$('#suggestbutton').click(function(){
  $('#learningrate').val(0.01)
  $('#inputdropoutrate').val(1.0)
  $('#outputdropoutrate').val(0.8)
  $('#timestamp').val(5)
  $('#sizelayer').val(32)
  $('#initialmoney').val(10000)
  $('#maxbuy').val(5)
  $('#maxsell').val(10)
  $('#epoch').val(10)
  $('#history').val(4)
  $('#future').val(30)
  $('#smooth').val(0.5)
})
$('#suggestbutton').click()
$('#trainbutton').click(function(){
  $('#log').html('');
  $('#log-invest').html('');
  $('.close-first').css('display','block');
  if(parseFloat($('#inputdropoutrate').val())<0 || parseFloat($('#inputdropoutrate').val())>1){
    Materialize.toast('input dropout must bigger than 0 and less than 1', 4000)
    return
  }
  if(parseFloat($('#smooth').val())<0 || parseFloat($('#smooth').val())>1){
    Materialize.toast('smoothing weights must bigger than 0 and less than 1', 4000)
    return
  }
  if(parseFloat($('#outputdropoutrate').val())<0 || parseFloat($('#outputdropoutrate').val())>1){
    Materialize.toast('output dropout must bigger than 0 and less than 1', 4000)
    return
  }
  setTimeout(function(){
    minmax_scaled = minmax_1d(close);
    timestamp = parseInt($('#timestamp').val())
    epoch = parseInt($('#epoch').val())
    future = parseInt($('#future').val())
    X_scaled = minmax_scaled.scaled.slice([0],[Math.floor(minmax_scaled.scaled.shape[0]/timestamp)*timestamp+1])
    cells = [tf.layers.lstmCell({units: parseInt($('#sizelayer').val())})];
    rnn = tf.layers.rnn({cell: cells, returnSequences: true,returnState:true});
    dense_layer = tf.layers.dense({units: 1, activation: 'linear'});
    function f(x,states){
      x = dropout_nn(x,parseFloat($('#inputdropoutrate').val()))
      forward = rnn.apply(x,{initialState:states});
      last_sequences = dropout_nn(forward[0].reshape([x.shape[1],parseInt($('#sizelayer').val())]),parseFloat($('#outputdropoutrate').val()))
      return {'forward':dense_layer.apply(last_sequences),'states_1':forward[1],'states_2':forward[2]}
    }
    cost = (label, pred) => tf.square(tf.sub(label,pred)).mean();
    optimizer = tf.train.adam(parseFloat($('#learningrate').val()));
    batch_states = [tf.zeros([1,parseInt($('#sizelayer').val())]),tf.zeros([1,parseInt($('#sizelayer').val())])];
    arr_loss = [], arr_layer = []
    function async_training_loop(callback) {
      (function loop(i) {
        var total_loss = 0
        for(var k = 0; k < Math.floor(X_scaled.shape[0]/timestamp)*timestamp; k+=timestamp){
          batch_x = X_scaled.slice([k],[timestamp]).reshape([1,-1,1])
          batch_y = X_scaled.slice([k+1],[timestamp]).reshape([-1,1])
          feed = f(batch_x,batch_states)
          optimizer.minimize(() => cost(batch_y,f(batch_x,batch_states)['forward']));
          total_loss += parseFloat(cost(batch_y,f(batch_x,batch_states)['forward']).toString().slice(7));
          batch_states = [feed.states_1,feed.states_2]
        }
        total_loss /= Math.floor(X_scaled.shape[0]/timestamp);
        arr_loss.push(total_loss)
        output_predict = nj.zeros([X_scaled.shape[0]+future, 1])
        output_predict.slice([0,1],null).assign(tf_str_tolist(X_scaled.slice(0,1))[0],false)
        upper_b = Math.floor(X_scaled.shape[0]/timestamp)*timestamp
        distance_upper_b = X_scaled.shape[0] - upper_b
        batch_states = [tf.zeros([1,parseInt($('#sizelayer').val())]),tf.zeros([1,parseInt($('#sizelayer').val())])];
        for(var k = 0; k < (Math.floor(X_scaled.shape[0]/timestamp)*timestamp); k+=timestamp){
          batch_x = X_scaled.slice([k],[timestamp]).reshape([1,-1,1])
          feed = f(batch_x,batch_states)
          state_forward = tf_nj_list(feed.forward)
          output_predict.slice([k+1,k+1+timestamp],null).assign(state_forward,false)
          batch_states = [feed.states_1,feed.states_2]
        }
        batch_x = X_scaled.slice([upper_b],[distance_upper_b]).reshape([1,-1,1])
        feed = f(batch_x,batch_states)
        state_forward = tf_nj_list(feed.forward)
        output_predict.slice([upper_b+1,X_scaled.shape[0]+1],null).assign(state_forward,false)
        pointer = X_scaled.shape[0]+1
        tensor_output_predict = output_predict.reshape([-1]).tolist()
        batch_states = [feed.states_1,feed.states_2]
        for(var k = 0; k < future-1; k+=1){
          batch_x = tf.tensor(tensor_output_predict.slice(pointer-timestamp,pointer)).reshape([1,-1,1])
          feed = f(batch_x,batch_states)
          state_forward = tf_nj_list(feed.forward.transpose())
          tensor_output_predict[pointer] = state_forward[0][4]
          pointer += 1
          batch_states = [feed.states_1,feed.states_2]
        }
        $('#log').append('Epoch: '+(i+1)+', avg loss: '+total_loss+'<br>');
        predicted_val = tf_nj_list_flatten(reverse_minmax_1d(tf.tensor(tensor_output_predict),minmax_scaled['min'],minmax_scaled['max']))
        predicted_val = smoothing_line(predicted_val,parseFloat($('#smooth').val()))
        $('#div_output').attr('style','height:450px;');
        new_date = stock_date.slice()
        for(var k = 0; k < future; k+=1){
          somedate = new Date(new_date[new_date.length-1])
          somedate.setDate(somedate.getDate() + 1)
          dd = somedate.getDate()
          mm = somedate.getMonth() + 1
          y = somedate.getFullYear()
          new_date.push(y.toString()+'-'+mm.toString()+'-'+dd.toString())
        }

        option = {
          animation: false,
          color: color_list,
          title: {
            left: 'center'
          },
          legend: {
            top: 30,
            data: ['STOCK', 'MA5', 'MA10', 'MA20', 'MA30','predicted close']
          },
          tooltip: {
            trigger: 'axis',
            position: function (pt) {
              return [pt[0], '10%'];
            }
          },
          axisPointer: {
            link: [{
              xAxisIndex: [0, 1]
            }]
          },
          dataZoom: [{
            type: 'slider',
            xAxisIndex: [0, 1],
            realtime: false,
            start: 0,
            end: 100,
            top: 65,
            height: 20,
            handleIcon: 'M10.7,11.9H9.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4h1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
            handleSize: '120%'
          }, {
            type: 'inside',
            xAxisIndex: [0, 1],
            start: 40,
            end: 70,
            top: 30,
            height: 20
          }],
          xAxis: [{
            type: 'category',
            data: new_date,
            boundaryGap : false,
            axisLine: { lineStyle: { color: '#777' } },
            axisLabel: {
              formatter: function (value) {
                return echarts.format.formatTime('MM-dd', value);
              }
            },
            min: 'dataMin',
            max: 'dataMax',
            axisPointer: {
              show: true
            }
          }, {
            type: 'category',
            gridIndex: 1,
            data: stock_date,
            scale: true,
            boundaryGap : false,
            splitLine: {show: false},
            axisLabel: {show: false},
            axisTick: {show: false},
            axisLine: { lineStyle: { color: '#777' } },
            splitNumber: 20,
            min: 'dataMin',
            max: 'dataMax',
            axisPointer: {
              type: 'shadow',
              label: {show: false},
              triggerTooltip: true,
              handle: {
                show: true,
                margin: 30,
                color: '#B80C00'
              }
            }
          }],
          yAxis: [{
            scale: true,
            splitNumber: 2,
            axisLine: { lineStyle: { color: '#777' } },
            splitLine: { show: true },
            axisTick: { show: false },
            axisLabel: {
              inside: true,
              formatter: '{value}\n'
            }
          }, {
            scale: true,
            gridIndex: 1,
            splitNumber: 2,
            axisLabel: {show: false},
            axisLine: {show: false},
            axisTick: {show: false},
            splitLine: {show: false}
          }],
          grid: [{
            left: 20,
            right: 20,
            top: 110,
          }, {
            left: 20,
            right: 20,
            top: 400
          }],
          graphic: [{
            type: 'group',
            left: 'center',
            top: 70,
            width: 300,
            bounding: 'raw',
            children: [{
              id: 'MA5',
              type: 'text',
              style: {fill: color_list[1]},
              left: 0
            }, {
              id: 'MA10',
              type: 'text',
              style: {fill: color_list[2]},
              left: 'center'
            }, {
              id: 'MA20',
              type: 'text',
              style: {fill: color_list[3]},
              right: 0
            }]
          }],
          series: [{
            name: 'Volume',
            type: 'bar',
            xAxisIndex: 1,
            yAxisIndex: 1,
            itemStyle: {
              normal: {
                color: '#7fbe9e'
              },
              emphasis: {
                color: '#140'
              }
            },
            data: volume
          }, {
            type: 'candlestick',
            name: 'STOCK',
            data: stocks,
            itemStyle: {
              normal: {
                color: '#ef232a',
                color0: '#14b143',
                borderColor: '#ef232a',
                borderColor0: '#14b143'
              },
              emphasis: {
                color: 'black',
                color0: '#444',
                borderColor: 'black',
                borderColor0: '#444'
              }
            }
          }, {
            name: 'MA5',
            type: 'line',
            data: dataMA5,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            }
          }, {
            name: 'MA10',
            type: 'line',
            data: dataMA10,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            }
          }, {
            name: 'MA20',
            type: 'line',
            data: dataMA20,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            }
          },
          {
            name: 'MA30',
            type: 'line',
            data: dataMA30,
            smooth: true,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 1
              }
            }
          },
          {
            name: 'predicted close',
            type: 'line',
            data: predicted_val,
            smooth: false,
            showSymbol: false,
            lineStyle: {
              normal: {
                width: 2
              }
            }
          }]
        };

        var chart_stock = echarts.init(document.getElementById('div_output'));
        chart_stock.setOption(option,true);
        calculate_distribution(close,predicted_val)
        option = {
          title:{
            text:'loss graph'
          },
          xAxis: {
            type: 'category',
            data: arange(0,arr_loss.length,1)
          },
          yAxis: {
            type: 'value'
          },
          grid:{
            bottom:'10%'
          },
          series: [{
            data: arr_loss,
            type: 'line'
          }]
        };
        var chart_line = echarts.init(document.getElementById('div_loss'));
        chart_line.setOption(option,true);
        if (i < (epoch-1)) {
          setTimeout(function() {loop(++i)}, 2000);
        } else {
          callback();
        }
      }(0));
    }
    async_training_loop(function() {
      $('#log').append('Done training!');
      my_investment = simple_investor(close,predicted_val,parseInt($('#history').val()),
      parseFloat($('#initialmoney').val()),parseInt($('#maxbuy').val()),parseInt($('#maxsell').val()),new_date)
      $('#table-body').html('');
      for(var i = 0; i < my_investment['output'].length; i++) $('#table-body').append(my_investment['output'][i]);
      $('#log-invest').append("<h6 class='header'>Overall gain: "+my_investment['overall gain']+", Overall investment: "+my_investment['overall investment']+"%</h5>")
      total_investment = my_investment['overall investment']
      total_gain = my_investment['overall gain']
      stock_changes = predicted_val[predicted_val.length-1] - close[0]
      stock_changes_percent = (stock_changes / close[0])*100

      var markpoints = []
      for (var i = 0; i < my_investment['buy_X'].length;i++){
        ind = new_date.indexOf(my_investment['buy_X'][i])
        markpoints.push({name: 'buy', value: 'buy', xAxis: ind, yAxis: my_investment['buy_Y'][i],itemStyle:{color:'#61a0a8'}})
      }
      for (var i = 0; i < my_investment['sell_X'].length;i++){
        ind = new_date.indexOf(my_investment['sell_X'][i])
        markpoints.push({name: 'sell', value: 'sell', xAxis: ind, yAxis: my_investment['sell_Y'][i],itemStyle:{color:'#c23531'}})
      }
      option = {
        animation: false,
        color: color_list,
        title: {
          left: 'center'
        },
        legend: {
          top: 30,
          data: ['STOCK', 'MA5', 'MA10', 'MA20', 'MA30','predicted close','sell','buy']
        },
        tooltip: {
          trigger: 'axis',
          position: function (pt) {
            return [pt[0], '10%'];
          }
        },
        axisPointer: {
          link: [{
            xAxisIndex: [0, 1]
          }]
        },
        dataZoom: [{
          type: 'slider',
          xAxisIndex: [0, 1],
          realtime: false,
          start: 0,
          end: 100,
          top: 65,
          height: 20,
          handleIcon: 'M10.7,11.9H9.3c-4.9,0.3-8.8,4.4-8.8,9.4c0,5,3.9,9.1,8.8,9.4h1.3c4.9-0.3,8.8-4.4,8.8-9.4C19.5,16.3,15.6,12.2,10.7,11.9z M13.3,24.4H6.7V23h6.6V24.4z M13.3,19.6H6.7v-1.4h6.6V19.6z',
          handleSize: '120%'
        }, {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 40,
          end: 70,
          top: 30,
          height: 20
        }],
        xAxis: [{
          type: 'category',
          data: new_date,
          boundaryGap : false,
          axisLine: { lineStyle: { color: '#777' } },
          axisLabel: {
            formatter: function (value) {
              return echarts.format.formatTime('MM-dd', value);
            }
          },
          min: 'dataMin',
          max: 'dataMax',
          axisPointer: {
            show: true
          }
        }, {
          type: 'category',
          gridIndex: 1,
          data: stock_date,
          scale: true,
          boundaryGap : false,
          splitLine: {show: false},
          axisLabel: {show: false},
          axisTick: {show: false},
          axisLine: { lineStyle: { color: '#777' } },
          splitNumber: 20,
          min: 'dataMin',
          max: 'dataMax',
          axisPointer: {
            type: 'shadow',
            label: {show: false},
            triggerTooltip: true,
            handle: {
              show: true,
              margin: 30,
              color: '#B80C00'
            }
          }
        }],
        yAxis: [{
          scale: true,
          splitNumber: 2,
          axisLine: { lineStyle: { color: '#777' } },
          splitLine: { show: true },
          axisTick: { show: false },
          axisLabel: {
            inside: true,
            formatter: '{value}\n'
          }
        }, {
          scale: true,
          gridIndex: 1,
          splitNumber: 2,
          axisLabel: {show: false},
          axisLine: {show: false},
          axisTick: {show: false},
          splitLine: {show: false}
        }],
        grid: [{
          left: 20,
          right: 20,
          top: 110,
        }, {
          left: 20,
          right: 20,
          top: 400
        }],
        graphic: [{
          type: 'group',
          left: 'center',
          top: 70,
          width: 300,
          bounding: 'raw',
          children: [{
            id: 'MA5',
            type: 'text',
            style: {fill: color_list[1]},
            left: 0
          }, {
            id: 'MA10',
            type: 'text',
            style: {fill: color_list[2]},
            left: 'center'
          }, {
            id: 'MA20',
            type: 'text',
            style: {fill: color_list[3]},
            right: 0
          }]
        }],
        series: [{
          name: 'Volume',
          type: 'bar',
          xAxisIndex: 1,
          yAxisIndex: 1,
          itemStyle: {
            normal: {
              color: '#7fbe9e'
            },
            emphasis: {
              color: '#140'
            }
          },
          data: volume
        }, {
          type: 'candlestick',
          name: 'STOCK',
          data: stocks,
          markPoint: {
            data: markpoints
          },
          itemStyle: {
            normal: {
              color: '#ef232a',
              color0: '#14b143',
              borderColor: '#ef232a',
              borderColor0: '#14b143'
            },
            emphasis: {
              color: 'black',
              color0: '#444',
              borderColor: 'black',
              borderColor0: '#444'
            }
          }
        }, {
          name: 'MA5',
          type: 'line',
          data: dataMA5,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            normal: {
              width: 1
            }
          }
        }, {
          name: 'MA10',
          type: 'line',
          data: dataMA10,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            normal: {
              width: 1
            }
          }
        }, {
          name: 'MA20',
          type: 'line',
          data: dataMA20,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            normal: {
              width: 1
            }
          }
        },
        {
          name: 'MA30',
          type: 'line',
          data: dataMA30,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            normal: {
              width: 1
            }
          }
        },
        {
          name: 'predicted close',
          type: 'line',
          data: predicted_val,
          smooth: false,
          showSymbol: false,
          lineStyle: {
            normal: {
              width: 2
            }
          }
        }]
      };

      var chart_stock = echarts.init(document.getElementById('div_output'));
      chart_stock.setOption(option,true);
      // $('#after-hell').css('display','block');
      // formData = new FormData();
      // formData.append("date", JSON.stringify(stock_date));
      // formData.append("close", JSON.stringify(close));
      // formData.append("rolling", $('#history').val());
      //
      // xmlhttp = new XMLHttpRequest();
      // xmlhttp.onreadystatechange = function() {
      //   if (xmlhttp.readyState == 4 && xmlhttp.status == 200) {
      //     try{
      //       data = JSON.parse(this.responseText);
      //       plot_pairplot(data)
      //     }
      //     catch(err){
      //       Materialize.toast("error, unable to do post-processing, please try with different data",3000);
      //       return;
      //     }
      //     if(data['error']){
      //       Materialize.toast("error, unable to do post-processing, please try with different data",3000);
      //       return;
      //     }
      //     else{
      //     }
      //   }
      // };
      // xmlhttp.open("POST", "http://huseinhouse.com:8070/uploader", true);
      // xmlhttp.send(formData);

    });
  }, 500);
})

function plot_pairplot(val){
  var chart = echarts.init(document.getElementById('pairplot'));
  var columns = ['Close','Crude Oil','Diesel', 'Gasoline', 'Gold', 'Heating Oil', 'Kerosene', 'Natural Gas', 'Propane'];

  var grids = [];
  var xAxes = [];
  var yAxes = [];
  var series = [];
  var titles = [];
  var count = 0;
  for(var k = 8; k >= 0;k--){
    grids.push({
      show: false,
      borderWidth: 0,
      backgroundColor: '#fff',
      shadowColor: 'rgba(0, 0, 0, 0.3)',
      shadowBlur: 2
    });
    if(k%9==0){
      xAxes.push({
        type: 'category',
        name:'sell',
        show:true,
        data: val['pairplot'][k][0][0],
        gridIndex: count
      })
      xAxes.push({
        type: 'category',
        name:'buy',
        show:false,
        data: val['pairplot'][k][1][0],
        gridIndex: count
      })
      yAxes.push({
        type: 'value',
        show:true,
        gridIndex: count
      })
      series.push({
        data: val['pairplot'][k][0][1],
        name:'sell',
        type: 'bar',
        xAxisIndex: count,
        yAxisIndex: count,
      })
      series.push({
        data: val['pairplot'][k][1][1],
        name:'buy',
        type: 'bar',
        xAxisIndex: count,
        yAxisIndex: count,
      })
      titles.push({
        textAlign: 'center',
        text: columns[k]+' histogram',
        textStyle: {
          fontSize: 12,
          fontWeight: 'normal'
        }
      })
    }
    else{
      titles.push({
        textAlign: 'center',
        text: columns[0]+' vs '+columns[k],
        textStyle: {
          fontSize: 12,
          fontWeight: 'normal'
        }
      })
      xAxes.push({
        type: 'value',
        gridIndex: count,
        show: true,
        min:'dataMin',
        max:'dataMax'
      })
      yAxes.push({
        type: 'value',
        show: true,
        gridIndex: count,
        min:'dataMin',
        max:'dataMax'
      })
      series.push({
        data: val['pairplot'][k][0][0].map(function(el, idx) {
          return [el,val['pairplot'][k][0][1][idx]];
        }),
        type: 'scatter',
        name:'sell',
        xAxisIndex: count,
        yAxisIndex: count,
      })
      series.push({
        data: val['pairplot'][k][1][0].map(function(el, idx) {
          return [el,val['pairplot'][k][1][1][idx]];
        }),
        type: 'scatter',
        name:'buy',
        xAxisIndex: count,
        yAxisIndex: count,
      })
    }
    count++;
  }

  var rowNumber = Math.ceil(Math.sqrt(count));
  echarts.util.each(grids, function (grid, idx) {
    grid.left = ((idx % rowNumber) / rowNumber * 100 + 2) + '%';
    grid.top = (Math.floor(idx / rowNumber) / rowNumber * 93 + 15) + '%';
    grid.width = (1 / rowNumber * 100 - 5) + '%';
    grid.height = (1 / rowNumber * 90 -11) + '%';

    titles[idx].left = parseFloat(grid.left) + parseFloat(grid.width) / 2 + '%';
    titles[idx].top = (parseFloat(grid.top)-5) + '%';
  });

  option = {
    color:['#c23531', '#61a0a8'],
    legend: {
      data:['sell','buy'],
      top:'5%'
    },
    title: titles.concat([{
      text: 'Pairplot Study',
      top: 'top',
      left: 'center'
    }]),
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        animation: false
      }
    },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series: series
  };
  chart.setOption(option)


  var chart_pi = echarts.init(document.getElementById('pi_correlation'));
  var seriesData = [];
  var selected = {};
  for(var i = 0; i < val['pi'].length;i++){
    seriesData.push({'name':columns[i+1],'value':val['pi'][i]})
    selected[columns[i+1]]=true
  }
  option = {
    title : {
      text: 'Correlation Piechart',
      x:'center'
    },
    tooltip : {
      trigger: 'item',
      formatter: "{a} <br/>{b} : {c} ({d}%)"
    },
    legend: {
      type: 'scroll',
      orient: 'vertical',
      right: 10,
      top: 20,
      bottom: 20,
      data: columns.slice(1),
      selected: selected
    },
    series : [
      {
        name: 'correlation',
        type: 'pie',
        radius : '55%',
        center: ['40%', '50%'],
        data: seriesData,
        itemStyle: {
          emphasis: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  };
  chart_pi.setOption(option)

  option = {
    legend: {},
    tooltip: {
      trigger: 'axis',
      showContent: false
    },
    dataset: {
      source: val['data_stack']
    },
    xAxis: {type: 'category'},
    yAxis: {gridIndex: 0},
    grid: {top: '60%'},
    series: [
      {type: 'line', smooth: true, seriesLayoutBy: 'row'},
      {type: 'line', smooth: true, seriesLayoutBy: 'row'},
      {type: 'line', smooth: true, seriesLayoutBy: 'row'},
      {type: 'line', smooth: true, seriesLayoutBy: 'row'},
      {
        type: 'pie',
        id: 'pie',
        radius: '30%',
        center: ['50%', '35%'],
        label: {
          formatter: '{b}: {@2017-07-31} ({d}%)'
        },
        encode: {
          itemName: 'data',
          value: '2017-07-31',
          tooltip: '2017-07-31'
        }
      }
    ]
  };
  var chart_pie = echarts.init(document.getElementById('div_pie'));
  chart_pie.on('updateAxisPointer', function (event) {
    var xAxisInfo = event.axesInfo[0];
    if (xAxisInfo) {
      var dimension = xAxisInfo.value + 1;
      chart_pie.setOption({
        series: {
          id: 'pie',
          label: {
            formatter: '{b}: {@[' + dimension + ']} ({d}%)'
          },
          encode: {
            value: dimension,
            tooltip: dimension
          }
        }
      });
    }
  });
  chart_pie.setOption(option);

  var grids = [];
  var xAxes = [];
  var yAxes = [];
  var series = [];
  var titles = [];
  var count = 0;
  for(var i = 0; i < val['movement_changes'].length; i++){
    var data = [];
    for (var k = 0; k < val['movement_changes'][i]['movement'].length; k++) {
      data.push([val['movement_changes'][i]['date'][k],val['movement_changes'][i]['movement'][k]])
    }
    grids.push({
      show: true,
      borderWidth: 0,
      backgroundColor: '#fff',
      shadowColor: 'rgba(0, 0, 0, 0.3)',
      shadowBlur: 2
    });
    xAxes.push({
      type: 'category',
      show: false,
      min:'dataMin',
      max:'dataMax',
      gridIndex: count
    });
    yAxes.push({
      type: 'value',
      show: false,
      min:'dataMin',
      max:'dataMax',
      gridIndex: count
    });
    if(val['pct_changes'][i] < 0) color_graph = 'red';
    else color_graph = 'green';
    series.push({
      type: 'line',
      xAxisIndex: count,
      yAxisIndex: count,
      data: data,
      itemStyle:{
        color:color_graph
      },
      showSymbol: false,
      animationDuration: 1000
    });
    titles.push({
      textAlign: 'center',
      text: val['movement_changes'][i]['title']+' '+(val['pct_changes'][i]).toFixed(2)+'%',
      textStyle: {
        fontSize: 12,
        fontWeight: 'normal'
      }
    });
    count++;
  }
  var rowNumber = Math.ceil(Math.sqrt(count));
  echarts.util.each(grids, function (grid, idx) {
    grid.left = ((idx % rowNumber) / rowNumber * 100 + 0.5) + '%';
    grid.top = (Math.floor(idx / rowNumber) / rowNumber * 120 + 10) + '%';
    grid.width = (1 / rowNumber * 100 - 1) + '%';
    grid.height = (1 / rowNumber * 120 - 1) + '%';

    titles[idx].left = parseFloat(grid.left) + parseFloat(grid.width) / 2 + '%';
    titles[idx].top = parseFloat(grid.top) + '%';
  });

  option = {
    title: titles.concat([{
      text: 'Weekly % changes',
      top: 'top',
      left: 'center'
    }]),
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        animation: false
      }
    },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series: series
  };
  var chart_changes = echarts.init(document.getElementById('percent_changes'));
  chart_changes.setOption(option)
  generate_investment(['total investment(%): ','total gains: ','stock changes: ','stock changes (%): ','gold changes(%): ','crude oil changes(%): '],
[total_investment.toFixed(2), total_gain.toFixed(2), stock_changes.toFixed(2),
  stock_changes_percent.toFixed(2),(val['gain_crude_oil']*100).toFixed(2),(val['gain_gold']*100).toFixed(2)])
}
