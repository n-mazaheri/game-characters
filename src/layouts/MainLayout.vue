<template>
  <q-layout view="hHh lpR fFf">

    <q-header elevated class="bg-primary text-white">
      <q-toolbar>
        <q-btn dense flat round icon="menu" @click="toggleLeftDrawer" />

        <q-toolbar-title>
          <q-avatar square>
            <img src="../assets/1451303.png">
          </q-avatar>
          Game Characters
        </q-toolbar-title>
      </q-toolbar>
    </q-header>

    <q-drawer show-if-above v-model="leftDrawerOpen" side="left" elevated>
      <q-list class="q-pt-xl">
        <q-item clickable v-ripple v-on:click="f3">
          <q-item-section avatar>
            <q-icon name="home" size="md" />
          </q-item-section>
          <q-item-section class="text-weight-bold">Home</q-item-section>
        </q-item>
         <q-item clickable v-ripple v-on:click="f1">
          <q-item-section avatar>
             <q-avatar square>
            <img src="../assets/c11.png">
          </q-avatar>
          </q-item-section>
          <q-item-section class="text-weight-bold">Kleed</q-item-section>
        </q-item>
         <q-item clickable v-ripple v-on:click="f2">
          <q-item-section avatar>
             <q-avatar square>
            <img src="../assets/c22.png">
          </q-avatar>
          </q-item-section>
         <q-item-section class="text-weight-bold">Max</q-item-section>
        </q-item> </q-list>
    </q-drawer>

    <q-page-container>
      <router-view :f="f"/> 
    </q-page-container>

  </q-layout>
</template>

<script>
import { ref } from 'vue'
import { defineComponent } from 'vue'

export default defineComponent({
name: 'MainLayout',
   data() {
  return{  
      leftDrawerOpen:false,
      f:"New User"
   }
  },
  methods: {
  f1(){
       this.f="Kleed"
      },
      f2(){
      
      this.f="Max"
      },
       f3(){
      
      this.f="New User"
      },
 toggleLeftDrawer:function () {
        this.leftDrawerOpen= !this.leftDrawerOpen
      }
  }

})


/*{


  setup () {
    const leftDrawerOpen = ref(false)
    var f=ref(null)

    return {
      leftDrawerOpen,
      f1(){
       f="Kleed"
      },
      f2(){
      f="Max"
      },
      toggleLeftDrawer () {
        leftDrawerOpen.value = !leftDrawerOpen.value
      }
    }
  }
}
*/
</script>