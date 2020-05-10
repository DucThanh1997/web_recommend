<template>
  <div class="content">
    <h1 style="text-align: center"> Hệ hỗ trợ ra quyết định </h1>
    <h2>1. Lựa chọn thuật toán </h2>
    <div class="md-layout-item md-small-size-100 md-size-33" style="font-size: 25px">
      <md-field>
        <label for="movie">Khoa</label>
        <md-select v-model="khoa" name="movie" id="movie">
          <md-option value="toan_tin">Toán Tin</md-option>
          <md-option value="kinh_te">Kinh tế</md-option>
          <md-option value="ngon_ngu">Ngôn ngữ</md-option>
        </md-select>
      </md-field>
    </div>
    <br>
    <h2>2. Lựa chọn khoa </h2>
    
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
    <h2>3. Chọn tập dữ liệu đầu vào </h2>
    <div class="md-layout-item md-small-size-100 md-size-33">
      <md-field>
        <label>Chọn tệp tải lên</label>
      <md-file v-model="single" @change="selectedFile"/>
      </md-field>
    </div>
    <div class="table-gen">
        <md-button type="button" @click="genTable" class="md-button md-raised md-success md-theme-default">
          Dữ liệu sau khi được làm sạch
        </md-button>
        <p id="showData"></p>
    </div>
    <p class="para"> Những môn bị thừa: {{unecessary_subject}}</p>
    <p class="para"> Những môn bị thiếu: {{incompliance_subject}}</p>
    <md-button type="button" @click="onSubmit" class="md-button md-raised md-success md-theme-default">Dự đoán</md-button>

    <br>
    <h2 style="margin-top: 20px;" >4. Kết quả </h2>
    <p style="margin-top:10px" class="para">Dự đoán xếp loại tốt nghiệp: {{toan_tin}} với độ chính xác là {{score}}%</p>
      <!-- <p style="margin-top:10px">Dự đoán xếp loại tốt nghiệp TE: {{kinh_te}} với độ chính xác là {{score}}%</p>
      <p style="margin-top:10px">Dự đoán xếp loại tốt nghiệp TC: {{ngon_ngu}} với độ chính xác là {{score}}%</p>
      <p style="margin-top:10px">Dự đoán xếp loại tốt nghiệp TM: {{y_te}} với độ chính xác là {{score}}%</p> -->
    
    <br>
    <h2 style="margin-top: 20px;" >5. Lời khuyên </h2>
    <p class="para"> Theo dự đoán của chúng tôi bạn nên học: {{recommend}}</p>
  </div>
</template>

<script>
import Vue from "vue";
import axios from "axios";
export default {
    data() {
        return {
            file: "",
            thuat_toan: '',
            khoa: '',
            toan_tin: '',
            kinh_te: '',
            ngon_ngu: '',
            y_te: '',
            xa_hoi: '',
            row: [],
            header: [],
            score: 0,
            recommend: [],
            "unecessary_subject": "",
            "incompliance_subject": "",
        }
    },

    methods:{
        selectedFile(event){
            this.files = event.target.files[0]
            console.log("name: ", this.files.name)
            let formData = new FormData()
            formData.append('resource_csv',this.files, this.files.name)
            formData.append('thuat_toan', this.thuat_toan)
            formData.append('khoa', this.khoa)
            axios.post('http://127.0.0.1:5000/preprocess', formData).then((response) => {
                this.row = response.data.ordered_result_list,
                this.header = response.data.predict_subjects,
                this.unecessary_subject = response.data.unecessary_subject,
                this.incompliance_subject = response.data.incompliance_subject,
                console.log("unecessary_subject: ", this.unecessary_subject)
                console.log("incompliance_subject: ", this.incompliance_subject)
                console.log("this.row: ", this.row)
                console.log("this.header: ", this.header)
            })
            
        },
        genTable() {
            var table = document.createElement("table");

            // CREATE HTML TABLE HEADER ROW USING THE EXTRACTED HEADERS ABOVE.

            var tr = table.insertRow(-1);                   // TABLE ROW.
            console.log("this.header: ", this.header)
            for (var i = 0; i < this.header.length; i++) {
                var th = document.createElement("th");      // TABLE HEADER.
                th.innerHTML = this.header[i];
                tr.appendChild(th);
            }

            // ADD JSON DATA TO THE TABLE AS ROWS.
            tr = table.insertRow(-1);

            for (var i = 0; i < this.row.length; i++) {
                var tabCell = tr.insertCell(-1);
                tabCell.innerHTML = this.row[i];
            }
            console.log("bảng điểm: ", this.row)
            // FINALLY ADD THE NEWLY CREATED TABLE WITH JSON DATA TO A CONTAINER.
            var divContainer = document.getElementById("showData");
            divContainer.innerHTML = "";
            divContainer.appendChild(table);
        },

        onSubmit(){
            let formData = new FormData()
            formData.append('resource_csv',this.files, this.files.name)
            formData.append('thuat_toan', this.thuat_toan)
            formData.append('khoa', this.khoa)
            axios.post('http://127.0.0.1:5000/predict-overall', formData).then(response=>{
                this.toan_tin = response.data.toan_tin
                this.score = response.data.score
                this.recommend = response.data.recommend
            })
        }
    }
}
</script>

<style>
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

    /* overflow: scroll; */
}

  th, td, p, input {
      font:14px Verdana;
  }
  table {
    border: solid 1px #DDD;  
    border-spacing: 30px;
    border-collapse: "separate";
    overflow: scroll;
  }
   th, td 
  {
      border: solid 1px #DDD;
      border-collapse: collapse;
      padding: 5px 3px;
      text-align: center;
  }
  th {
      font-weight:bold;
  }
  .para {
    font-size: 20px !important;
    margin-top:20px !important
 }
</style>
