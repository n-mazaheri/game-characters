<template>
   <q-form
      @submit="onSubmit"
      @reset="onReset"
      class="q-gutter-md"
    >
    <div> {{f}}</div>

     <div class="row">
      <div class="col q-pr-sm">
         <q-input
        filled
        v-model="Race"
        label="Race"
      
      />
      </div>
      <div class="col">
        <q-input
        filled
        v-model="Name"
        label="Name"
   
      />
      </div>
    </div>
     <div class="row">
      <div class="col q-pr-sm">
         <q-input
        filled
        type="number"
        v-model="Height"
        label="Height"
  
      />
     
      </div>
      <div class="col">
     <q-input
        filled
        type="number"
        v-model="Power"
        label="Power"
    step="any"
      />
      </div>
    </div>
    
      <div class="row">
      <div class="col q-pr-sm">
        <q-input
        filled
        type="number"
        v-model="Magic"
        label="Magic Level"
     
      />
    
      </div>
      <div class="col">
           <q-input
        filled
        type="number"
        v-model="Greed"
        label="Greed Level"
  
      />
      </div>
    </div>

      <q-toggle v-model="AutoFarm" label="AutoFarm" />
 
     

      <div>
        <q-btn label="Submit" type="submit" color="primary" />
        <q-btn label="Reset" type="reset" color="primary" flat class="q-ml-sm" />
      </div>
    </q-form>
 
</template>
<script>
import { useQuasar } from 'quasar'
import { ref } from 'vue'
import { defineComponent } from 'vue'
import { api } from 'boot/axios'


export default  defineComponent({
name: 'MyForm', props: ['f'],
   data() {
  return{  
      Race:null,
      Name:null,
      Height:null,
      Power:null,
      Magic:null,
      Greed:null,
      AutoFarm:null,

     
   }
  },
  updated:function () {
 var str=null;
 if(this.f=="Kleed")
str='/a980d3fd-7eac-4450-a2fb-aa3be3ee01d6'
else if(this.f=="Max")
str="/ac05988f-e2e5-42f9-8088-9da6979926e3"
else if(this.f=="New User")
str=null;

if(str!=null)
{
    api.get(str)
      .then((response) => {
       response.data.map((data) =>{
       if(data.id=="race")
       this.Race=data.value
       if(data.id=="name")
       this.Name=data.value
       if(data.id=="height")
       this.Height=data.value
       if(data.id=="power")
       this.Power=data.value
       if(data.id=="magic-level")
       this.Magic=data.value
       if(data.id=="greed-level")
       this.Greed=data.value
       if(data.id=="autofarm")
       this.AutoFarm=data.value

       })
      
      })
      .catch(() => {
       console.log("error");
      })
      }
      else{
      
      this.Race=null
      this.Name=null
      this.Height=null
      this.Power=null
      this.Magic=null
      this.Greed=null
      this.AutoFarm=null  
      }
  },
 
  methods: {
  onReset(){
    this.Race=null
      this.Name=null
      this.Height=null
      this.Power=null
      this.Magic=null
      this.Greed=null
      this.AutoFarm=null  
  }
  }

})

</script>