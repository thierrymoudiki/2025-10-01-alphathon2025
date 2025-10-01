from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Pricing Dashboard")

# Serve templates and static files
templates = Jinja2Templates(directory="templates")
#app.mount("/static", StaticFiles(directory="static"), name="static")

import numpy as np
from typing import Dict, Optional

class PricingService:
    @staticmethod
    def calculate_lapse_probability(cumulative_returns: np.ndarray, lapse_params: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate lapse probabilities based on cumulative returns with optional customization.
        :param cumulative_returns: np.ndarray, the cumulative returns
        :param lapse_params: Optional[Dict], optional customization for lapse probabilities
        :return: np.ndarray, lapse probabilities
        """
        adj_prob = np.zeros_like(cumulative_returns)

        # Default lapse behavior params
        lapse_defaults = {
            'heavy_loss': (-0.10, 0.15, 2),  # Threshold, base prob, scale factor for heavy loss
            'light_loss': (0, 0.05, 1),     # Threshold, base prob, scale factor for light loss
            'high_gain': (0.20, 0.08, 0),    # Threshold, fixed prob for high gain
            'normal': (None, 0.05, 0)        # Normal range, fixed prob
        }

        # Merge with custom parameters if provided
        if lapse_params:
            lapse_defaults.update(lapse_params)

        # Apply performance-adjusted lapse probabilities
        mask_heavy_loss = cumulative_returns < lapse_defaults['heavy_loss'][0]
        mask_light_loss = (cumulative_returns < lapse_defaults['light_loss'][0]) & ~mask_heavy_loss
        mask_high_gain = cumulative_returns > lapse_defaults['high_gain'][0]
        mask_normal = ~mask_heavy_loss & ~mask_light_loss & ~mask_high_gain
        
        adj_prob[mask_heavy_loss] = lapse_defaults['heavy_loss'][1] + (lapse_defaults['heavy_loss'][0] - cumulative_returns[mask_heavy_loss]) * lapse_defaults['heavy_loss'][2]
        adj_prob[mask_light_loss] = lapse_defaults['light_loss'][1] + (-cumulative_returns[mask_light_loss]) * lapse_defaults['light_loss'][2]
        adj_prob[mask_high_gain] = lapse_defaults['high_gain'][1]
        adj_prob[mask_normal] = lapse_defaults['normal'][1]

        # Bound probabilities and convert to daily probabilities
        adj_prob = np.clip(adj_prob, 0.01, 0.20)
        daily_prob = 1 - (1 - adj_prob) ** (1/260)  # Assuming 260 trading days per year
        return daily_prob

    @staticmethod
    def price_participating_product(sims_data: np.ndarray, S0: float = 100, 
                                  participation_rate: float = 0.85, 
                                  risk_free_rate: float = 0.02, 
                                  n_periods: int = 30, 
                                  lapse_params: Optional[Dict] = None, 
                                  discount_factor_type: str = 'annual') -> Dict:
        """
        Vectorized pricing function for a participating product.
        :param sims_data: np.ndarray, simulated data for asset prices
        :param S0: float, initial asset price
        :param participation_rate: float, participation rate for positive returns
        :param risk_free_rate: float, the risk-free rate
        :param n_periods: int, number of periods for simulation
        :param lapse_params: Optional[Dict], custom lapse parameters (for flexibility)
        :param discount_factor_type: str, type of discount factor ('annual', 'quarterly', 'monthly')
        :return: Dict, dictionary with the price and statistics
        """
        n_paths = sims_data.shape[1]

        # Initialize matrices
        account_values = np.full((n_periods + 1, n_paths), S0)
        cashflows = np.zeros((n_periods + 1, n_paths))
        lapse_indicators = np.zeros((n_periods + 1, n_paths))

        # Simulate the asset and lapse events for all paths in a vectorized manner
        for t in range(n_periods):
            current_values = account_values[t, :]
            period_returns = sims_data[t, :]

            # Apply participation (85% of positive returns)
            credited_returns = np.where(period_returns > 0, 
                                         period_returns * participation_rate, 
                                         period_returns)

            # Update account values
            account_values[t + 1, :] = current_values * (1 + credited_returns)

            # Calculate cumulative returns (based on the new account values)
            cumulative_returns = (account_values[t + 1, :] / S0) - 1

            # Calculate lapse probabilities based on cumulative returns
            lapse_probs = PricingService.calculate_lapse_probability(cumulative_returns, lapse_params)

            # Determine lapses: vectorized check if random draw is below lapse probability
            lapse_events = (np.random.random(n_paths) < lapse_probs)

            # Apply lapse event: zero out account values for lapses after the event
            lapse_indicators[t + 1, lapse_events] = 1
            cashflows[t + 1, lapse_events] = account_values[t + 1, lapse_events]
            account_values[t + 1:, lapse_events] = 0

        # Final cashflows for non-lapsed paths (using vectorized mask)
        final_lapses = lapse_indicators[n_periods, :] == 0
        cashflows[n_periods, final_lapses] = account_values[n_periods, final_lapses]

        # Discount and calculate present value
        time_points = np.arange(n_periods + 1)
        
        # Adjust discounting based on chosen type (e.g., 'annual', 'quarterly', 'monthly')
        if discount_factor_type == 'annual':
            discount_factors = np.exp(-risk_free_rate * time_points / 260)
        elif discount_factor_type == 'quarterly':
            discount_factors = np.exp(-risk_free_rate * time_points / 65)
        elif discount_factor_type == 'monthly':
            discount_factors = np.exp(-risk_free_rate * time_points / 12)
        else:
            raise ValueError("Invalid discount factor type. Choose 'annual', 'quarterly', or 'monthly'.")

        # Calculate present value of cashflows
        pv_matrix = cashflows * discount_factors.reshape(-1, 1)
        path_pvs = np.sum(pv_matrix, axis=0)

        return {
            "price": float(np.mean(path_pvs)),
            "account_values": account_values.tolist(),
            "statistics": {
                "expected_lapse_rate": float(np.mean(np.sum(lapse_indicators, axis=0) > 0)),
                "final_value_mean": float(np.mean(account_values[n_periods, :])),
                "total_paths": n_paths,
                "price_distribution": path_pvs.tolist()
            }
        }


class DataFetcher:
    @staticmethod
    async def fetch_simulation_data(ticker: str, test_index: int) -> np.ndarray:
        """Fetch simulation data from the external API"""
        api_url = "https://pretrainedridge2f-8aee3d9572cc.herokuapp.com/forecast"
        data = {"ticker": ticker, "n_ahead": 30}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        # Parse simulations - convert list of lists to numpy array                        
                        sims_matrix = np.array(result.get("sims", []))
                        logger.info(f"Received data from API. Status:  {response.status}")
                        return sims_matrix
                    else:
                        logger.error(f"API request failed with status {response.status}")
                        # Return random data as fallback
                        return np.random.normal(0.0005, 0.02, (30, 250))
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            # Return random data as fallback
            return np.random.normal(0.0005, 0.02, (30, 250))

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    """Serve the main dashboard page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws/pricing")
async def websocket_pricing(websocket: WebSocket):
    """WebSocket endpoint for real-time pricing stream"""
    await manager.connect(websocket)
    pricing_service = PricingService()
    
    try:
        test_index = 1
        while test_index <= 20:
            # Fetch new data
            sims_data = await DataFetcher.fetch_simulation_data("DAX", test_index)
            
            # Calculate pricing
            pricing_results = pricing_service.price_participating_product(
                sims_data=sims_data,
                S0=100,
                participation_rate=0.85,
                risk_free_rate=0.02
            )
            
            # Prepare data for frontend
            account_vals = np.array(pricing_results["account_values"])
            time_points = np.arange(account_vals.shape[0])
            lower_bound = np.percentile(account_vals, 5, axis=1)
            upper_bound = np.percentile(account_vals, 95, axis=1)
            median_vals = np.median(account_vals, axis=1)
            
            # Price distribution data
            price_dist = pricing_results["statistics"]["price_distribution"]
            hist, bin_edges = np.histogram(price_dist, bins="auto", density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Send update to frontend
            await manager.send_personal_message(json.dumps({
                "update_count": test_index,
                "price": pricing_results["price"],
                "statistics": pricing_results["statistics"],
                "account_evolution": {
                    "time_points": time_points.tolist(),
                    "lower_bound": lower_bound.tolist(),
                    "upper_bound": upper_bound.tolist(),
                    "median_vals": median_vals.tolist()
                },
                "price_distribution": {
                    "bin_centers": bin_centers.tolist(),
                    "density": hist.tolist()
                }
            }), websocket)
            
            test_index += 1
            await asyncio.sleep(1)  # Wait xx seconds
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)