#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <deque>
#include <random>
#include <algorithm>
#include <sstream>
#include "math/simd_math.h"
#include "memory/allocator.h"
#include "gpu/vertex_manager.h"
#include "render/pipeline.h"
#include "profiler/profiler.h"

using namespace engine::math;
using namespace engine::gpu;
using namespace engine::render;
using namespace engine::profiler;

// ═══════════════════════════════════════════════
// Market Data Structures
// ═══════════════════════════════════════════════

enum class OrderSide { BUY, SELL };
enum class OrderType { MARKET, LIMIT };

struct Order {
    int         id;
    OrderSide   side;
    OrderType   type;
    double      price;
    int         quantity;
    double      timestamp;
    int         algoId;      // which algorithm placed it
};

struct Trade {
    double price;
    int    quantity;
    double timestamp;
};

struct PriceBar {
    double open, high, low, close;
    int    volume;
    double timestamp;
};

// Order book (simplified)
struct OrderBook {
    std::vector<Order> bids;   // sorted descending by price
    std::vector<Order> asks;   // sorted ascending by price
    
    double getBestBid() const { return bids.empty() ? 0.0 : bids.front().price; }
    double getBestAsk() const { return asks.empty() ? 0.0 : asks.front().price; }
    double getMidPrice() const { return (getBestBid() + getBestAsk()) / 2.0; }
    double getSpread() const { return getBestAsk() - getBestBid(); }
};

// ═══════════════════════════════════════════════
// Trading Algorithms
// ═══════════════════════════════════════════════

enum class AlgoType {
    MARKET_MAKER,      // Provides liquidity, profits from spread
    MOMENTUM,          // Follows trends
    MEAN_REVERSION,    // Bets on price returning to average
    ARBITRAGE,         // Exploits price discrepancies
    RANDOM             // Noise traders
};

struct TradingAlgo {
    int        id;
    AlgoType   type;
    std::string name;
    Vec3       color;          // for visualization
    
    // State
    double     cash;
    int        position;       // shares held (can be negative for short)
    double     pnl;            // profit/loss
    
    // Stats
    int        ordersPlaced;
    int        tradesExecuted;
    
    // Visual
    NodeHandle node;
    
    TradingAlgo(int id, AlgoType type, std::string name, Vec3 color)
        : id(id), type(type), name(name), color(color),
          cash(100000.0), position(0), pnl(0.0),
          ordersPlaced(0), tradesExecuted(0), node(InvalidNode) {}
};

// ═══════════════════════════════════════════════
// Market Simulator
// ═══════════════════════════════════════════════

class MarketSimulator {
public:
    MarketSimulator(double initialPrice = 100.0)
        : currentTime(0.0), currentPrice(initialPrice),
          nextOrderId(1), tradeCount(0)
    {
        priceHistory.push_back(initialPrice);
    }
    
    void step(double dt, std::vector<TradingAlgo>& algos, std::mt19937& rng) {
        currentTime += dt;
        
        // Each algo generates orders based on its strategy
        for (auto& algo : algos) {
            generateOrders(algo, rng);
        }
        
        // Match orders (simplified crossing)
        matchOrders(algos);
        
        // Random price walk (market noise)
        std::normal_distribution<> noise(0.0, 0.05);
        currentPrice += noise(rng);
        currentPrice = std::max(currentPrice, 1.0);  // price floor
        
        priceHistory.push_back(currentPrice);
        if (priceHistory.size() > 500) priceHistory.pop_front();
        
        // Update algos' PnL
        for (auto& algo : algos) {
            algo.pnl = algo.cash + algo.position * currentPrice - 100000.0;
        }
    }
    
    double getCurrentPrice() const { return currentPrice; }
    double getCurrentTime() const { return currentTime; }
    const std::deque<double>& getPriceHistory() const { return priceHistory; }
    const OrderBook& getOrderBook() const { return orderBook; }
    int getTotalTrades() const { return tradeCount; }
    
private:
    void generateOrders(TradingAlgo& algo, std::mt19937& rng) {
        std::uniform_real_distribution<> prob(0.0, 1.0);
        
        // Skip if not active this tick
        if (prob(rng) > 0.3) return;
        
        double mid = orderBook.getMidPrice();
        if (mid == 0.0) mid = currentPrice;
        
        Order order;
        order.id = nextOrderId++;
        order.algoId = algo.id;
        order.timestamp = currentTime;
        
        switch (algo.type) {
            case AlgoType::MARKET_MAKER: {
                // Place limit orders on both sides
                if (prob(rng) > 0.5) {
                    // Bid below mid
                    order.side = OrderSide::BUY;
                    order.type = OrderType::LIMIT;
                    order.price = mid - 0.05 - prob(rng) * 0.1;
                    order.quantity = 100 + (int)(prob(rng) * 400);
                    orderBook.bids.push_back(order);
                } else {
                    // Ask above mid
                    order.side = OrderSide::SELL;
                    order.type = OrderType::LIMIT;
                    order.price = mid + 0.05 + prob(rng) * 0.1;
                    order.quantity = 100 + (int)(prob(rng) * 400);
                    orderBook.asks.push_back(order);
                }
                algo.ordersPlaced++;
                break;
            }
            
            case AlgoType::MOMENTUM: {
                // Follow recent trend
                double trend = priceHistory.size() > 10 
                    ? (currentPrice - priceHistory[priceHistory.size()-10]) 
                    : 0.0;
                
                if (trend > 0.2) {
                    // Buy on uptrend
                    order.side = OrderSide::BUY;
                    order.type = OrderType::MARKET;
                    order.price = currentPrice * 1.01;  // willing to pay premium
                    order.quantity = 50 + (int)(prob(rng) * 150);
                    orderBook.bids.push_back(order);
                    algo.ordersPlaced++;
                } else if (trend < -0.2) {
                    // Sell on downtrend
                    order.side = OrderSide::SELL;
                    order.type = OrderType::MARKET;
                    order.price = currentPrice * 0.99;
                    order.quantity = 50 + (int)(prob(rng) * 150);
                    orderBook.asks.push_back(order);
                    algo.ordersPlaced++;
                }
                break;
            }
            
            case AlgoType::MEAN_REVERSION: {
                // Calculate simple moving average
                double sma = 0.0;
                int window = std::min(20, (int)priceHistory.size());
                for (int i = 0; i < window; ++i) {
                    sma += priceHistory[priceHistory.size() - 1 - i];
                }
                sma /= window;
                
                double deviation = currentPrice - sma;
                
                if (deviation > 0.5) {
                    // Price too high, sell
                    order.side = OrderSide::SELL;
                    order.type = OrderType::LIMIT;
                    order.price = currentPrice - 0.05;
                    order.quantity = 100 + (int)(prob(rng) * 200);
                    orderBook.asks.push_back(order);
                    algo.ordersPlaced++;
                } else if (deviation < -0.5) {
                    // Price too low, buy
                    order.side = OrderSide::BUY;
                    order.type = OrderType::LIMIT;
                    order.price = currentPrice + 0.05;
                    order.quantity = 100 + (int)(prob(rng) * 200);
                    orderBook.bids.push_back(order);
                    algo.ordersPlaced++;
                }
                break;
            }
            
            case AlgoType::ARBITRAGE: {
                // Look for spread opportunities
                double spread = orderBook.getSpread();
                if (spread > 0.2) {
                    // Wide spread, provide liquidity
                    if (prob(rng) > 0.5) {
                        order.side = OrderSide::BUY;
                        order.price = orderBook.getBestBid() + 0.01;
                    } else {
                        order.side = OrderSide::SELL;
                        order.price = orderBook.getBestAsk() - 0.01;
                    }
                    order.type = OrderType::LIMIT;
                    order.quantity = 200 + (int)(prob(rng) * 300);
                    
                    if (order.side == OrderSide::BUY)
                        orderBook.bids.push_back(order);
                    else
                        orderBook.asks.push_back(order);
                    algo.ordersPlaced++;
                }
                break;
            }
            
            case AlgoType::RANDOM: {
                // Noise trader
                order.side = prob(rng) > 0.5 ? OrderSide::BUY : OrderSide::SELL;
                order.type = OrderType::MARKET;
                order.price = currentPrice * (prob(rng) > 0.5 ? 1.02 : 0.98);
                order.quantity = 10 + (int)(prob(rng) * 90);
                
                if (order.side == OrderSide::BUY)
                    orderBook.bids.push_back(order);
                else
                    orderBook.asks.push_back(order);
                algo.ordersPlaced++;
                break;
            }
        }
    }
    
    void matchOrders(std::vector<TradingAlgo>& algos) {
        // Sort order book
        std::sort(orderBook.bids.begin(), orderBook.bids.end(),
            [](const Order& a, const Order& b) { return a.price > b.price; });
        std::sort(orderBook.asks.begin(), orderBook.asks.end(),
            [](const Order& a, const Order& b) { return a.price < b.price; });
        
        // Match crossing orders
        while (!orderBook.bids.empty() && !orderBook.asks.empty()) {
            Order& bestBid = orderBook.bids.front();
            Order& bestAsk = orderBook.asks.front();
            
            if (bestBid.price >= bestAsk.price) {
                // Trade occurs
                int qty = std::min(bestBid.quantity, bestAsk.quantity);
                double price = (bestBid.price + bestAsk.price) / 2.0;
                
                // Update algos
                auto& buyer = algos[bestBid.algoId];
                auto& seller = algos[bestAsk.algoId];
                
                buyer.cash -= price * qty;
                buyer.position += qty;
                buyer.tradesExecuted++;
                
                seller.cash += price * qty;
                seller.position -= qty;
                seller.tradesExecuted++;
                
                tradeCount++;
                currentPrice = price;  // last trade sets price
                
                // Update quantities
                bestBid.quantity -= qty;
                bestAsk.quantity -= qty;
                
                if (bestBid.quantity == 0) orderBook.bids.erase(orderBook.bids.begin());
                if (bestAsk.quantity == 0) orderBook.asks.erase(orderBook.asks.begin());
            } else {
                break;  // no more matches
            }
        }
        
        // Clean up stale orders (older than 5 seconds)
        auto isStale = [this](const Order& o) { return currentTime - o.timestamp > 5.0; };
        orderBook.bids.erase(std::remove_if(orderBook.bids.begin(), orderBook.bids.end(), isStale),
                             orderBook.bids.end());
        orderBook.asks.erase(std::remove_if(orderBook.asks.begin(), orderBook.asks.end(), isStale),
                             orderBook.asks.end());
    }
    
    double currentTime;
    double currentPrice;
    int    nextOrderId;
    int    tradeCount;
    
    std::deque<double> priceHistory;
    OrderBook orderBook;
};

// ═══════════════════════════════════════════════
// 3D Visualization
// ═══════════════════════════════════════════════

MeshHandle createCube(float size) {
    auto& vm = VertexManager::instance();
    
    MeshDescriptor desc;
    desc.name = "Cube";
    desc.vertexFormat = VertexFormatType::PositionNormalUV;
    desc.boundsMin = {-size, -size, -size};
    desc.boundsMax = { size,  size,  size};
    
    MeshHandle handle = vm.createMesh(desc);
    Mesh* mesh = vm.getMesh(handle);
    
    // Simple cube vertices
    VertexPositionNormalUV verts[] = {
        // Front
        {{-size,-size, size}, {0,0,1}, {0,0}}, {{ size,-size, size}, {0,0,1}, {1,0}},
        {{ size, size, size}, {0,0,1}, {1,1}}, {{-size, size, size}, {0,0,1}, {0,1}},
        // Back
        {{-size,-size,-size}, {0,0,-1}, {1,0}}, {{-size, size,-size}, {0,0,-1}, {1,1}},
        {{ size, size,-size}, {0,0,-1}, {0,1}}, {{ size,-size,-size}, {0,0,-1}, {0,0}},
        // Top
        {{-size, size,-size}, {0,1,0}, {0,1}}, {{-size, size, size}, {0,1,0}, {0,0}},
        {{ size, size, size}, {0,1,0}, {1,0}}, {{ size, size,-size}, {0,1,0}, {1,1}},
        // Bottom
        {{-size,-size,-size}, {0,-1,0}, {1,1}}, {{ size,-size,-size}, {0,-1,0}, {0,1}},
        {{ size,-size, size}, {0,-1,0}, {0,0}}, {{-size,-size, size}, {0,-1,0}, {1,0}},
        // Right
        {{ size,-size,-size}, {1,0,0}, {1,0}}, {{ size, size,-size}, {1,0,0}, {1,1}},
        {{ size, size, size}, {1,0,0}, {0,1}}, {{ size,-size, size}, {1,0,0}, {0,0}},
        // Left
        {{-size,-size,-size}, {-1,0,0}, {0,0}}, {{-size,-size, size}, {-1,0,0}, {1,0}},
        {{-size, size, size}, {-1,0,0}, {1,1}}, {{-size, size,-size}, {-1,0,0}, {0,1}}
    };
    
    uint32_t indices[] = {
        0,1,2, 0,2,3,       // front
        4,5,6, 4,6,7,       // back
        8,9,10, 8,10,11,    // top
        12,13,14, 12,14,15, // bottom
        16,17,18, 16,18,19, // right
        20,21,22, 20,22,23  // left
    };
    
    mesh->vertexBuffer().upload(verts, 24);
    mesh->indexBuffer().upload(indices, 36);
    
    return handle;
}

// ═══════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════

int main() {
    std::cout << "═══════════════════════════════════════════════════════\n";
    std::cout << "  QUANTITATIVE TRADING ALGORITHM SIMULATOR\n";
    std::cout << "  Real-Time Market Visualization (NYSE-Style)\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    auto& pipeline = RenderPipeline::instance();
    auto& vm = VertexManager::instance();
    std::mt19937 rng(42);
    
    // ─────────────────────────────────────────────
    // Setup
    // ─────────────────────────────────────────────
    const double initialPrice = 100.0;
    MarketSimulator market(initialPrice);
    
    // Create trading algorithms
    std::vector<TradingAlgo> algos = {
        {0, AlgoType::MARKET_MAKER,   "Market Maker A",  {0.2f, 0.8f, 0.2f}},
        {1, AlgoType::MARKET_MAKER,   "Market Maker B",  {0.3f, 0.9f, 0.3f}},
        {2, AlgoType::MOMENTUM,       "Momentum Trader", {0.9f, 0.2f, 0.2f}},
        {3, AlgoType::MEAN_REVERSION, "Mean Reversion",  {0.2f, 0.4f, 0.9f}},
        {4, AlgoType::ARBITRAGE,      "Arbitrageur",     {0.9f, 0.8f, 0.2f}},
        {5, AlgoType::RANDOM,         "Noise Trader 1",  {0.5f, 0.5f, 0.5f}},
        {6, AlgoType::RANDOM,         "Noise Trader 2",  {0.6f, 0.6f, 0.6f}},
    };
    
    // ─────────────────────────────────────────────
    // Rendering Setup
    // ─────────────────────────────────────────────
    pipeline.setCamera(
        {0, 15, 30},   // eye
        {0, 5, 0},     // target
        {0, 1, 0},     // up
        60.0f, 16.0f/9.0f, 0.1f, 500.0f
    );
    
    // Lighting
    Light keyLight;
    keyLight.type = LightType::Directional;
    keyLight.direction = Vec3(-0.5f, -1.0f, -0.5f).normalized();
    keyLight.color = {1.0f, 1.0f, 0.95f};
    keyLight.intensity = 1.5f;
    pipeline.addLight(keyLight);
    
    Light fillLight;
    fillLight.type = LightType::Directional;
    fillLight.direction = Vec3(0.5f, -0.5f, 0.5f).normalized();
    fillLight.color = {0.5f, 0.6f, 0.8f};
    fillLight.intensity = 0.5f;
    pipeline.addLight(fillLight);
    
    // Create meshes
    MeshHandle cubeMesh = createCube(0.3f);
    
    // Create materials for each algo
    std::vector<MaterialHandle> materials;
    for (const auto& algo : algos) {
        PBRMaterial mat;
        mat.albedo = algo.color;
        mat.metallic = 0.3f;
        mat.roughness = 0.6f;
        materials.push_back(pipeline.createMaterial(mat));
    }
    
    // Create scene nodes for algos (positioned in a circle)
    for (size_t i = 0; i < algos.size(); ++i) {
        float angle = (float)i / algos.size() * TWO_PI;
        float radius = 8.0f;
        
        algos[i].node = pipeline.createNode(algos[i].name);
        SceneNode* node = pipeline.getNode(algos[i].node);
        node->mesh = cubeMesh;
        node->material = materials[i];
        node->position = {radius * std::cos(angle), 0.0f, radius * std::sin(angle)};
        node->dirty = true;
    }
    
    // Price history visualization (vertical bars)
    std::vector<NodeHandle> priceBarNodes;
    MaterialHandle priceBarMat;
    {
        PBRMaterial mat;
        mat.albedo = {0.1f, 0.7f, 1.0f};
        mat.metallic = 0.1f;
        mat.roughness = 0.4f;
        priceBarMat = pipeline.createMaterial(mat);
    }
    
    for (int i = 0; i < 100; ++i) {
        NodeHandle h = pipeline.createNode("PriceBar_" + std::to_string(i));
        SceneNode* node = pipeline.getNode(h);
        node->mesh = cubeMesh;
        node->material = priceBarMat;
        node->visible = false;  // hide until we have data
        priceBarNodes.push_back(h);
    }
    
    pipeline.setThreadCount(4);
    
    // ─────────────────────────────────────────────
    // Simulation Loop
    // ─────────────────────────────────────────────
    const int   totalFrames = 1000;
    const float deltaTime   = 0.1f;   // 10 ticks per second
    
    std::cout << "Simulating " << totalFrames << " frames (100 seconds of market activity)...\n\n";
    
    for (int frame = 0; frame < totalFrames; ++frame) {
        ScopedProfile frameProfile("Frame");
        
        // ── Market simulation ──
        {
            ScopedProfile simProfile("Market Simulation");
            market.step(deltaTime, algos, rng);
        }
        
        // ── Update visualizations ──
        {
            ScopedProfile vizProfile("Visualization Update");
            
            // Update algo node heights based on PnL
            for (auto& algo : algos) {
                SceneNode* node = pipeline.getNode(algo.node);
                if (node) {
                    float pnlRatio = algo.pnl / 10000.0f;  // normalize
                    node->position.y = pnlRatio * 3.0f;    // height represents profit/loss
                    node->scale = {1.0f, 1.0f + std::abs(pnlRatio) * 0.5f, 1.0f};
                    node->rotation.y += deltaTime * (0.5f + pnlRatio * 0.1f);
                    node->dirty = true;
                }
            }
            
            // Update price history bars
            const auto& history = market.getPriceHistory();
            int barCount = std::min((int)history.size(), (int)priceBarNodes.size());
            
            for (int i = 0; i < barCount; ++i) {
                int historyIdx = history.size() - barCount + i;
                double price = history[historyIdx];
                double normalizedPrice = (price - initialPrice) / initialPrice * 10.0f;  // scale
                
                SceneNode* node = pipeline.getNode(priceBarNodes[i]);
                node->visible = true;
                node->position = {(float)(i - barCount/2) * 0.2f, (float)normalizedPrice * 0.5f, -5.0f};
                node->scale = {0.08f, std::abs((float)normalizedPrice) * 0.5f + 0.1f, 0.08f};
                node->dirty = true;
            }
        }
        
        // ── Render pipeline ──
        {
            ScopedProfile renderProfile("Render");
            pipeline.updateSceneGraph();
            pipeline.frustumCull();
            pipeline.buildDrawCalls();
            pipeline.sortBatches();
            pipeline.submit();
        }
        
        // ── Print status every 100 frames ──
        if ((frame + 1) % 100 == 0) {
            std::cout << "\n━━━━━ Frame " << (frame + 1) << " ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
            std::cout << "Time: " << std::fixed << std::setprecision(1) << market.getCurrentTime() << "s\n";
            std::cout << "Price: $" << std::setprecision(2) << market.getCurrentPrice() << "\n";
            std::cout << "Spread: $" << market.getOrderBook().getSpread() << "\n";
            std::cout << "Total Trades: " << market.getTotalTrades() << "\n";
            std::cout << "Order Book: " << market.getOrderBook().bids.size() 
                      << " bids, " << market.getOrderBook().asks.size() << " asks\n\n";
            
            std::cout << "Algorithm Performance:\n";
            std::cout << std::setw(18) << "Name"
                      << std::setw(12) << "PnL"
                      << std::setw(10) << "Position"
                      << std::setw(10) << "Orders"
                      << std::setw(10) << "Trades" << "\n";
            std::cout << std::string(60, '-') << "\n";
            
            for (const auto& algo : algos) {
                std::cout << std::setw(18) << algo.name
                          << std::setw(12) << std::setprecision(2) << algo.pnl
                          << std::setw(10) << algo.position
                          << std::setw(10) << algo.ordersPlaced
                          << std::setw(10) << algo.tradesExecuted << "\n";
            }
        }
        
        Profiler::instance().frameEnd();
    }
    
    // ─────────────────────────────────────────────
    // Final Report
    // ─────────────────────────────────────────────
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "  SIMULATION COMPLETE\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    // Market statistics
    std::cout << "Market Statistics:\n";
    std::cout << "  Final Price: $" << std::fixed << std::setprecision(2) 
              << market.getCurrentPrice() << "\n";
    std::cout << "  Price Change: " << std::setprecision(1)
              << ((market.getCurrentPrice() - initialPrice) / initialPrice * 100.0) << "%\n";
    std::cout << "  Total Trades: " << market.getTotalTrades() << "\n";
    std::cout << "  Avg Spread: $" << std::setprecision(3) 
              << market.getOrderBook().getSpread() << "\n\n";
    
    // Rank algos by PnL
    std::vector<TradingAlgo> ranked = algos;
    std::sort(ranked.begin(), ranked.end(), 
        [](const TradingAlgo& a, const TradingAlgo& b) { return a.pnl > b.pnl; });
    
    std::cout << "Algorithm Leaderboard (by PnL):\n";
    std::cout << std::string(70, '=') << "\n";
    for (size_t i = 0; i < ranked.size(); ++i) {
        const auto& a = ranked[i];
        std::cout << (i+1) << ". " << std::setw(18) << std::left << a.name
                  << " | PnL: $" << std::setw(10) << std::right << std::fixed 
                  << std::setprecision(2) << a.pnl
                  << " | Position: " << std::setw(5) << a.position
                  << " | Win Rate: " << std::setprecision(1)
                  << (a.tradesExecuted > 0 ? (a.pnl > 0 ? 100.0 : 0.0) : 0.0) << "%\n";
    }
    std::cout << std::string(70, '=') << "\n\n";
    
    // Performance profiling
    Profiler::instance().printReport();
    
    const auto& stats = pipeline.stats();
    std::cout << "\nRendering Stats:\n";
    std::cout << "  Nodes rendered: " << stats.visibleNodes << "/" << stats.totalNodes << "\n";
    std::cout << "  Draw calls: " << stats.totalDrawCalls << "\n";
    std::cout << "  Batches: " << stats.totalBatches << "\n";
    std::cout << "  Instanced draw calls: " << stats.batchedDrawCalls << "\n";
    
    // Cleanup
    vm.destroyMesh(cubeMesh);
    for (auto& algo : algos) {
        pipeline.destroyNode(algo.node);
    }
    for (auto h : priceBarNodes) {
        pipeline.destroyNode(h);
    }
    
    return 0;
}