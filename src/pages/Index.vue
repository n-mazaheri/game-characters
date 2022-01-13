<template>
  <q-page class="flex flex-center">
<MyForm :f="this.f"/>
 </q-page>
</template>

<script>
import MyForm from "components/MyForm.vue"
import { defineComponent } from 'vue';

export default 
defineComponent({
props: ['f'],
  name: 'PageIndex',
  components:{MyForm}
})

</script>
