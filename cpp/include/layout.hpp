#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace tmap {

// =============================================================================
// Configuration
// =============================================================================

/// Placer algorithm for initial node placement during uncoarsening
enum class Placer {
    Barycenter,  // Place at barycenter of neighbors (default)
    Solar,       // Use solar merger info
    Circle,      // Place in circle around barycenter
    Median,      // Median position of neighbors
    Random,      // Random position (non-deterministic!)
    Zero         // Same position as representative
};

/// Merger algorithm for graph coarsening
enum class Merger {
    EdgeCover,        // Edge cover based
    LocalBiconnected, // Avoids distortions (default)
    Solar,            // Solar system partitioning
    IndependentSet    // GRIP-style independent set
};

/// Scaling type for ScalingLayout
enum class ScalingType {
    Absolute,              // Absolute scaling factor
    RelativeToAvgLength,   // Scale relative to average edge weights
    RelativeToDesiredLength,
    RelativeToDrawing      // Scale relative to drawing (default)
};

/// Layout configuration matching original TMAP parameters
struct LayoutConfig {
    // FastMultipoleEmbedder
    int fme_iterations = 1000;
    int fme_precision = 4;

    // ScalingLayout
    int sl_repeats = 1;
    int sl_extra_scaling_steps = 2;
    double sl_scaling_min = 1.0;
    double sl_scaling_max = 1.0;
    ScalingType sl_scaling_type = ScalingType::RelativeToDrawing;

    // ModularMultilevelMixer
    int mmm_repeats = 1;

    // Placer/Merger
    Placer placer = Placer::Barycenter;
    Merger merger = Merger::LocalBiconnected;
    double merger_factor = 2.0;
    int merger_adjustment = 0;

    // Node size (affects repulsion)
    float node_size = 1.0f / 65.0f;

    // Determinism: if set, use single thread + seed
    bool deterministic = false;
    std::optional<uint32_t> seed = std::nullopt;
};

// =============================================================================
// Result types
// =============================================================================

/// Layout result: coordinates + edge topology
struct LayoutResult {
    std::vector<float> x;        // X coordinates (n_nodes)
    std::vector<float> y;        // Y coordinates (n_nodes)
    std::vector<uint32_t> s;     // Edge sources (n_edges)
    std::vector<uint32_t> t;     // Edge targets (n_edges)
};

// =============================================================================
// Layout functions
// =============================================================================

/// Compute layout from edge list
///
/// @param vertex_count Number of vertices
/// @param edges Edge list as (source, target, weight) tuples
/// @param config Layout configuration
/// @param create_mst If true, compute MST first (removes non-MST edges)
/// @return Layout result with coordinates and edge topology
LayoutResult layout_from_edge_list(
    uint32_t vertex_count,
    const std::vector<std::tuple<uint32_t, uint32_t, float>>& edges,
    const LayoutConfig& config = LayoutConfig{},
    bool create_mst = true
);

}  // namespace tmap
