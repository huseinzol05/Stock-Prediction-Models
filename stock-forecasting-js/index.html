<html>
<head>
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
  <link href="css/materialize.min.css" type="text/css" rel="stylesheet" media="screen,projection"/>
  <link href="css/style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
  <style>
  .close-first{
    display: none;
  }
  </style>
</head><br>
<div class="row" style="padding-left:10px;padding-right:10px">
  <ul class="collapsible" data-collapsible="accordion">
    <li>
      <div class="collapsible-header"><i class="material-icons" style="font-size:3rem">settings</i>
        <div class="row" style="margin-bottom:10px;margin-top:10px">
          <div class="col s3 m1">
            Settings
          </div>
          <div class="input-field col s12 m1 right" style="margin-top:5px; width:160px">
            <button id="trainbutton" class="waves-effect waves-light btn red lighten-2">Train</button>
          </div>
          <div class="input-field col s12 m1 right" style="margin-top:5px; width:160px">
            <button id="suggestbutton" class="waves-effect waves-light btn blue lighten-2">Suggest</button>
          </div>
          <div class="file-field input-field col s12 m1 right" style="margin-top:5px; width:160px">
            <div class="btn blue lighten-2" style="height:36px; line-height:2.5rem">
              <span>Pick CSV</span>
              <input id="uploadcsv" type="file">
            </div>
          </div>
        </div>
      </div>
      <div class="collapsible-body"><span>
        <div class="row center">
          <div class="input-field col m2 offset-m1" style="margin-left:5.33%">
            Neural Network settings
          </div>
          <div class="input-field col s12 m1">
            <input id="learningrate" type="number" placeholder="Eg: 0.001" class="validate tooltipped" data-position="bottom" data-delay="50" data-tooltip="learning rate during training">
            <label class="active">Learning rate</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="inputdropoutrate" type="number" placeholder="Eg: 0.9" class="validate tooltipped" data-position="bottom" data-delay="50" data-tooltip="dropout rate for LSTM input">
            <label class="active">Input dropout rate</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="outputdropoutrate" type="number" placeholder="Eg: 0.9" class="validate tooltipped" data-position="bottom" data-delay="50" data-tooltip="dropout rate for LSTM output">
            <label class="active">Output dropout rate</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="timestamp" type="number" class="validate tooltipped" placeholder="Eg: 5" data-position="bottom" data-delay="50" data-tooltip="Trends for every minibatch">
            <label class="active">Timestamp per training</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="sizelayer" type="number" class="validate tooltipped" placeholder="Eg: 64" data-position="bottom" data-delay="50" data-tooltip="LSTM size">
            <label class="active">Size layer</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="epoch" type="number" class="validate tooltipped" placeholder="Eg: 10" data-position="bottom" data-delay="50" data-tooltip="Total epoch">
            <label class="active">Training Iteration</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="future" type="number" class="validate tooltipped" placeholder="Eg: 10" data-position="bottom" data-delay="50" data-tooltip="number of days forecast">
            <label class="active">Future days to forecast</label>
          </div>
          <div class="input-field col s12 m1">
            <input id="smooth" type="number" class="validate tooltipped" placeholder="Eg: 10" data-position="bottom" data-delay="50" data-tooltip="Rate anchor smoothing for trends">
            <label class="active">Smoothing weights</label>
          </div>
        </div>
        <div class="row center">
          <div class="input-field col m2 offset-m1" style="margin-left:5.33%">
            Buying & Selling simulation
          </div>
          <div class="input-field col s12 m2">
            <input id="initialmoney" type="number" placeholder="Eg: 10000" class="validate tooltipped" data-position="bottom" data-delay="50" data-tooltip="Money in for simulation">
            <label class="active">Initial money(usd)</label>
          </div>
          <div class="input-field col s12 m2">
            <input id="maxbuy" type="number" placeholder="Eg: 5" class="validate tooltipped" data-position="bottom" data-delay="50" data-tooltip="Max unit to buy">
            <label class="active">Max buy(unit)</label>
          </div>
          <div class="input-field col s12 m2">
            <input id="maxsell" type="number" class="validate tooltipped" placeholder="Eg: 10" data-position="bottom" data-delay="50" data-tooltip="Max unit to sell">
            <label class="active">Max sell(unit)</label>
          </div>
          <div class="input-field col s12 m2">
            <input id="history" type="number" class="validate tooltipped" placeholder="Eg: 5" data-position="bottom" data-delay="50" data-tooltip="MA to compare of">
            <label class="active">Historical rolling</label>
          </div>
        </div>
      </span></div>
    </li>
  </ul>
</div>


<h6 class='header center light'>WARNING, This website may hang during training, and do not use this website to buy real stock!<br><br>Default stock is Google 2018, you can try upload any stock CSV</h6>
<div class="row" style="padding-left:10px;padding-right:10px">
  <div class="col s12 m12">
    <div id="div_output" style="height: 500px;"></div>
  </div>
</div>
<br>
<div class="row close-first" style="padding-left:10px;padding-right:10px">
  <div class="col s12 m8">
    <div id="div_dist" style="height: 450px;"></div>
  </div>
  <div class="col s12 m4">
    <div class="row">
      <div id="div_loss" style="height: 250px;"></div>
    </div>
    <div class="row" id="log" style="height: 150px; overflow:auto;">
    </div>
  </div>
</div>
<div class="row" style="padding-left:10px;padding-right:10px">
  <ul class="collapsible" data-collapsible="accordion">
    <li>
      <div class="collapsible-header"><i class="material-icons">archive</i>Simulation log</div>
      <div class="collapsible-body"><span>
        <table class="bordered highlight">
          <thead>
            <tr>
              <th>Date</th>
              <th>Action</th>
              <th>Price</th>
              <th>Investment</th>
              <th>Balance</th>
            </tr>
          </thead>
          <tbody id='table-body'>
          </tbody>
        </table><br>
        <span id="log-invest"></span>
      </span></div>
    </li>
  </ul>
</div>
<div class="row center" id="color-investment">
</div>
<script src="js/tf.js"></script>
<script src="js/jquery-3.3.1.min.js"></script>
<script src="js/materialize.min.js"></script>
<script src="js/d3.v3.min.js"></script>
<script src="js/numeric-1.2.6.min.js"></script>
<script src="js/numjs.min.js"></script>
<script src="js/utils.js"></script>
<script src="js/echarts.min.js"></script>
<script src="js/echarts-gl.min.js"></script>
<script src="js/papaparse.min.js"></script>
<script src="data/google.js"> </script>
<script src="init.js"> </script>
