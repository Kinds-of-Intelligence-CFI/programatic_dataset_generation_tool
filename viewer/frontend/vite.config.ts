import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "node:path";

// The build emits into ../static (served by the FastAPI app). Bundle files go
// under app/ rather than the Vite default assets/ so they never collide with
// the dataset's /assets/ route.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: resolve(__dirname, "../static"),
    emptyOutDir: true,
    assetsDir: "app",
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:7341",
      "/assets": "http://127.0.0.1:7341",
    },
  },
  test: {
    environment: "node",
  },
});
