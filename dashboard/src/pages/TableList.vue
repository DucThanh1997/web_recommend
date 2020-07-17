<template>
  <div class="content">
    <h1 style="text-align: center"> Bộ Huấn Luyện </h1>
    <br>
    <h2>1. Lựa chọn thuật toán </h2>
    <div class="md-layout-item md-small-size-100 md-size-33" style="font-size: 25px">
      <md-field>
        <label for="movie">Thuật toán</label>
        <md-select v-model="thuat_toan" name="movie" id="movie">
          <md-option value="knn">Knn</md-option>
          <md-option value="naive">NaiveBayes</md-option>
          <md-option value="ID3">ID3</md-option>
        </md-select>
      </md-field>
    </div>
    <br>
    <h2>2. Nhập các tham số đầu vào </h2>
    <div v-if="thuat_toan === 'knn'">
        <h3 style="margin-top:10px">- Chia tập training và testing </h3>
        <form>
            0 
            <input
                style="margin-top:10px"
                type="range"
                class="custom-range slider"
                id="customRange"
                name="points1"
                min="0"
                v-bind:max="max"
                v-bind:value="training"
                v-on:input="onChg($event)">
            100
        </form>
        <p style="" class="para">Huấn luyện: {{ training }} % </p>
        <p style="margin-top:1px" class="para">Thử nghiệm: {{ testing }} % </p>

        <label for="neighbour" class="para">Số láng giềng: </label>
        <md-input class="para"
            type="text" 
            v-model="neighbour">
        </md-input>

    </div>

    <div v-if="thuat_toan === 'ID3'">
        <h3 style="margin-top:10px">- Chia tập training và testing </h3>
        0 
        <input
            style="margin-top:10px"
            type="range"
            class="slider custom-range"
            id="customRange"
            min="0"
            v-bind:max="max"
            v-bind:value="training"
            v-on:input="onChg($event)">
          100
        <p style="margin-top:20px" class="para">Huấn luyện: {{ training }} % </p>
        <p style="margin-top:1px" class="para">Thử nghiệm: {{ testing }} % </p>

    </div>

    <div v-if="thuat_toan === 'naive'">
        <h3 style="margin-top:10px">- Chia tập training và testing </h3>
        0 
        <input
            type="range"
            class="slider"
            min="0"
            v-bind:max="max"

            v-bind:value="training"
            
            v-on:input="onChg($event)">
          100
        <p style="margin-top:20px" class="para">Huấn luyện: {{ training }} % </p>
        <p style="margin-top:1px" class="para">Thử nghiệm: {{ testing }} % </p>

    </div>
    <br>
    <h2>3. Nhập dữ liệu đầu vào </h2>
    <div class="md-layout-item md-small-size-100 md-size-33">
        <md-field>
          <label>Chọn tệp</label>
        <md-file v-model="single" @change="selectedFile"/>
        </md-field>
    </div>
    <md-button type="button" @click="onSubmit" class="md-button md-raised md-success md-theme-default">Huấn luyện</md-button>
    <br>
    <br>
    <br>
    <h2>4. Kết quả </h2>
    <p style="margin-top:10px; font-size:18px">Độ chính xác: {{score * 100}}%</p>
  </div>
</template>

<script>
import axios from "axios";
export default {
  data() {
    return {
      file: '',
      khoa: '',
      thuat_toan: '',
      score: 0,
      max: 100,
      min: 0,
      training: 0,
      testing: 0,
      neighbour: 0,
    }
  },
  methods:{
    selectedFile(event){
        console.log(event)
        this.files = event.target.files[0]
    },
    onChg (e) {
        this.training = e.target.value;
        this.testing = 100 - e.target.value
    },
    onSubmit(){
      let formData = new FormData()
      console.log("phan_phoi: ", this.phan_phoi)
      formData.append('resource_csv',this.files, this.files.name)
      formData.append('thuat_toan', this.thuat_toan)
      formData.append('khoa', this.khoa)
      formData.append('training_percent', this.training)
      formData.append('testing_percent', this.testing)
      formData.append('neighbour', this.neighbour)
      
      axios.post('http://127.0.0.1:5000/train-sample', formData).then(response=>{
          console.log("a1: ", response.data.score)
          this.score = response.data.score
      })
    }
  }
};
</script>

<style>
  .para {
    font-size: 20px !important;
    margin-top:20px !important
  }
 .stretch-card{
    height: calc(100vh - 64px);
    overflow-y: scroll;
  }
  .table-gen {
      overflow: scroll;
      padding: 3px;
  }
  .table-gen p {
    padding-top: 50px;

    overflow: scroll;
  }

  /* th, td, p, input {
    font:14px Verdana;
  } */


</style>
