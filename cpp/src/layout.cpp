#include "layout.hpp"

#include <algorithm>
#include <stdexcept>

// OGDF includes
#include <ogdf/basic/Graph.h>
#include <ogdf/basic/GraphAttributes.h>
#include <ogdf/basic/PreprocessorLayout.h>
#include <ogdf/basic/basic.h>
#include <ogdf/basic/extended_graph_alg.h>
#include <ogdf/basic/simple_graph_alg.h>
#include <ogdf/energybased/FastMultipoleEmbedder.h>
#include <ogdf/energybased/multilevel_mixer/BarycenterPlacer.h>
#include <ogdf/energybased/multilevel_mixer/CirclePlacer.h>
#include <ogdf/energybased/multilevel_mixer/EdgeCoverMerger.h>
#include <ogdf/energybased/multilevel_mixer/IndependentSetMerger.h>
#include <ogdf/energybased/multilevel_mixer/LocalBiconnectedMerger.h>
#include <ogdf/energybased/multilevel_mixer/MedianPlacer.h>
#include <ogdf/energybased/multilevel_mixer/ModularMultilevelMixer.h>
#include <ogdf/energybased/multilevel_mixer/MultilevelGraph.h>
#include <ogdf/energybased/multilevel_mixer/RandomPlacer.h>
#include <ogdf/energybased/multilevel_mixer/ScalingLayout.h>
#include <ogdf/energybased/multilevel_mixer/SolarMerger.h>
#include <ogdf/energybased/multilevel_mixer/SolarPlacer.h>
#include <ogdf/energybased/multilevel_mixer/ZeroPlacer.h>
#include <ogdf/graphalg/steiner_tree/EdgeWeightedGraph.h>
#include <ogdf/packing/ComponentSplitterLayout.h>
#include <ogdf/packing/TileToRowsCCPacker.h>

namespace tmap {

namespace {

// Convert our Placer enum to OGDF InitialPlacer
ogdf::InitialPlacer* create_placer(Placer p) {
    switch (p) {
        case Placer::Barycenter: return new ogdf::BarycenterPlacer();
        case Placer::Solar:      return new ogdf::SolarPlacer();
        case Placer::Circle:     return new ogdf::CirclePlacer();
        case Placer::Median:     return new ogdf::MedianPlacer();
        case Placer::Random:     return new ogdf::RandomPlacer();
        case Placer::Zero:       return new ogdf::ZeroPlacer();
        default:                 return new ogdf::BarycenterPlacer();
    }
}

// Convert our Merger enum to OGDF MultilevelBuilder
ogdf::MultilevelBuilder* create_merger(Merger m, double factor, int adjustment) {
    ogdf::MultilevelBuilder* merger = nullptr;
    switch (m) {
        case Merger::EdgeCover: {
            auto* ecm = new ogdf::EdgeCoverMerger();
            ecm->setFactor(factor);
            ecm->setEdgeLengthAdjustment(adjustment);
            merger = ecm;
            break;
        }
        case Merger::LocalBiconnected: {
            auto* lbm = new ogdf::LocalBiconnectedMerger();
            lbm->setFactor(factor);
            lbm->setEdgeLengthAdjustment(adjustment);
            merger = lbm;
            break;
        }
        case Merger::Solar: {
            merger = new ogdf::SolarMerger();
            break;
        }
        case Merger::IndependentSet: {
            auto* ism = new ogdf::IndependentSetMerger();
            ism->setSearchDepthBase(factor);
            merger = ism;
            break;
        }
        default: {
            auto* lbm = new ogdf::LocalBiconnectedMerger();
            lbm->setFactor(factor);
            lbm->setEdgeLengthAdjustment(adjustment);
            merger = lbm;
            break;
        }
    }
    return merger;
}

// Convert our ScalingType to OGDF
ogdf::ScalingLayout::ScalingType to_ogdf_scaling(ScalingType st) {
    switch (st) {
        case ScalingType::Absolute:
            return ogdf::ScalingLayout::ScalingType::Absolute;
        case ScalingType::RelativeToAvgLength:
            return ogdf::ScalingLayout::ScalingType::RelativeToAvgLength;
        case ScalingType::RelativeToDesiredLength:
            return ogdf::ScalingLayout::ScalingType::RelativeToDesiredLength;
        case ScalingType::RelativeToDrawing:
        default:
            return ogdf::ScalingLayout::ScalingType::RelativeToDrawing;
    }
}

}  // namespace

LayoutResult layout_from_edge_list(
    uint32_t vertex_count,
    const std::vector<std::tuple<uint32_t, uint32_t, float>>& edges,
    const LayoutConfig& config,
    bool create_mst
) {
    LayoutResult result;

    // Handle empty/trivial cases
    if (vertex_count == 0) {
        return result;
    }
    if (vertex_count == 1) {
        result.x = {0.0f};
        result.y = {0.0f};
        return result;
    }

    // Set seed for determinism if requested (default to 0 when deterministic with no seed)
    if (config.deterministic) {
        const int seed = static_cast<int>(config.seed.value_or(0u));
        ogdf::setSeed(seed);
    }

    // Build OGDF graph
    ogdf::EdgeWeightedGraph<float> g;
    std::vector<ogdf::node> nodes(vertex_count);

    for (uint32_t i = 0; i < vertex_count; i++) {
        nodes[i] = g.newNode();
    }

    // Find max weight for normalization
    float max_weight = 0.0f;
    for (const auto& [src, tgt, w] : edges) {
        if (w > max_weight) max_weight = w;
    }
    if (max_weight == 0.0f) max_weight = 1.0f;

    // Add edges (normalized, skip negatives)
    for (const auto& [src, tgt, w] : edges) {
        if (w >= 0.0f && src < vertex_count && tgt < vertex_count) {
            g.newEdge(nodes[src], nodes[tgt], w / max_weight);
        }
    }

    // Clean graph
    ogdf::makeLoopFree(g);
    ogdf::makeParallelFreeUndirected(g);

    // Count connected components
    ogdf::NodeArray<int> component(g);
    int n_components = ogdf::connectedComponents(g, component);

    // Compute MST if requested
    if (create_mst && g.numberOfEdges() > 0) {
        ogdf::EdgeArray<float> weights = g.edgeWeights();
        ogdf::makeMinimumSpanningTree(g, weights);
    }

    // Create GraphAttributes
    ogdf::GraphAttributes ga(g);
    ga.setAllHeight(config.node_size);
    ga.setAllWidth(config.node_size);

    // Create MultilevelGraph
    ogdf::MultilevelGraph mlg(ga);

    // Setup FastMultipoleEmbedder
    auto* fme = new ogdf::FastMultipoleEmbedder();
    fme->setNumIterations(config.fme_iterations);
    fme->setMultipolePrec(config.fme_precision);
    fme->setDefaultEdgeLength(1);
    fme->setDefaultNodeSize(1);

    if (config.deterministic) {
        fme->setRandomize(false);
        fme->setNumberOfThreads(1);
    } else {
        fme->setRandomize(false);  // Still default to false for consistency
    }

    // Setup ScalingLayout
    auto* sl = new ogdf::ScalingLayout();
    sl->setLayoutRepeats(config.sl_repeats);
    sl->setSecondaryLayout(fme);
    sl->setExtraScalingSteps(config.sl_extra_scaling_steps);
    sl->setScalingType(to_ogdf_scaling(config.sl_scaling_type));
    sl->setScaling(config.sl_scaling_min, config.sl_scaling_max);

    // Setup Placer and Merger
    ogdf::InitialPlacer* placer = create_placer(config.placer);
    ogdf::MultilevelBuilder* merger = create_merger(
        config.merger, config.merger_factor, config.merger_adjustment);

    // Setup ModularMultilevelMixer
    auto* mmm = new ogdf::ModularMultilevelMixer();
    mmm->setLayoutRepeats(config.mmm_repeats);
    mmm->setLevelLayoutModule(sl);
    mmm->setInitialPlacer(placer);
    mmm->setMultilevelBuilder(merger);

    // Run layout
    if (n_components > 1) {
        auto* csl = new ogdf::ComponentSplitterLayout();
        auto* packer = new ogdf::TileToRowsCCPacker();
        csl->setPacker(packer);
        csl->setLayoutModule(mmm);

        ogdf::PreprocessorLayout ppl;
        ppl.setLayoutModule(csl);
        ppl.setRandomizePositions(false);
        ppl.call(mlg);
    } else {
        mmm->call(mlg);
    }

    // Export coordinates
    mlg.exportAttributes(ga);

    result.x.resize(vertex_count);
    result.y.resize(vertex_count);

    int i = 0;
    for (ogdf::node v : g.nodes) {
        result.x[i] = static_cast<float>(ga.x(v));
        result.y[i] = static_cast<float>(ga.y(v));
        i++;
    }

    // Normalize to [-0.5, 0.5]
    if (!result.x.empty()) {
        float min_x = *std::min_element(result.x.begin(), result.x.end());
        float max_x = *std::max_element(result.x.begin(), result.x.end());
        float min_y = *std::min_element(result.y.begin(), result.y.end());
        float max_y = *std::max_element(result.y.begin(), result.y.end());

        float diff_x = max_x - min_x;
        float diff_y = max_y - min_y;

        // Avoid division by zero
        if (diff_x < 1e-10f) diff_x = 1.0f;
        if (diff_y < 1e-10f) diff_y = 1.0f;

        for (size_t j = 0; j < result.x.size(); j++) {
            result.x[j] = (result.x[j] - min_x) / diff_x - 0.5f;
            result.y[j] = (result.y[j] - min_y) / diff_y - 0.5f;
        }
    }

    // Extract edges
    for (ogdf::edge e : g.edges) {
        result.s.push_back(static_cast<uint32_t>(e->source()->index()));
        result.t.push_back(static_cast<uint32_t>(e->target()->index()));
    }

    return result;
}

}  // namespace tmap
