import { Platform } from 'react-native';

/**
 * CENTRAL BACKEND CONFIGURATION
 * ----------------------------
 * 1. LOCAL TESTING (Web Browser): Use 'http://localhost:8000'
 * 2. LOCAL TESTING (Physical Phone): Use your machine IP (e.g., 'http://192.168.137.235:8000')
 * 3. PRODUCTION: Use your Railway URL (e.g., 'https://englishessay-production.up.railway.app')
 */

const PRODUCTION_URL = "https://englishessay-production-6ce5.up.railway.app";

// YOUR CURRENT LOCAL IP (found via ipconfig)
const LOCAL_IP = "192.168.137.135";

// PRODUCTION - Using the Railway link as requested
export const BACKEND_URL = PRODUCTION_URL;

/* 
// LOCAL TESTING - Use this if you want to switch back to local testing
export const BACKEND_URL = Platform.select({
  web: "http://localhost:8000",
  default: `http://${LOCAL_IP}:8000`
});
*/

