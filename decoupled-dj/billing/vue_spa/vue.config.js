// const { defineConfig } = require('@vue/cli-service')

const path = require("path"); // newly added


// module.exports = defineConfig({
//   transpileDependencies: true
// })

module.exports = {
  publicPath: process.env.VUE_APP_STATIC_URL,
  outputDir: path.resolve(__dirname, "../static", "billing"),
  indexPath: path.resolve(__dirname, "../templates/", "billing", "index.html"),
  devServer: {
    proxy: "http://localhost:8000"
  }
};


