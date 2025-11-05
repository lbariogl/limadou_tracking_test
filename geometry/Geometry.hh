#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <array>

namespace Geometry
{
  // =============================
  // Pixel and Chip geometry
  // =============================

  static constexpr int PixelNCols = 1024;
  static constexpr int PixelNRows = 512;
  static constexpr double PixelSizeCols = 0.02924;     // mm
  static constexpr double PixelSizeRows = 0.02688;     // mm
  static constexpr double ChipDistanceX = 0.150;       // mm
  static constexpr double ChipDistanceY = 0.150;       // mm
  static constexpr double ChipSizeX = PixelSizeCols * PixelNCols; // 29.94 mm
  static constexpr double ChipSizeY = PixelSizeRows * PixelNRows; // 13.76 mm
  static constexpr double ChipStaveDistanceY = 7.22312; // mm

  // =============================
  // Tracker 1 (TR1)
  // =============================

  static constexpr double TR1Thickness = 2.0; // mm
  static constexpr std::array<double, 3> TR1Size = {154.6, 32.5, TR1Thickness};
  static constexpr double TR1CenterZ = 0.0;
  static constexpr double TR1GapY = 1.9;

  // =============================
  // Tracker 2 (TR2)
  // =============================

  static constexpr double TR2Thickness = 8.0; // mm
  static constexpr std::array<double, 3> TR2Size = {36.0, 150.0, TR2Thickness};
  static constexpr double TR2CenterZ = TR1CenterZ + 60.5;
  static constexpr double TR2GapX = 2.0;

  // =============================
  // Stave (Z positions)
  // =============================

  static constexpr std::array<double, 3> StaveZ = {
      17.825 + TR1CenterZ,
      17.825 + 8.5 + TR1CenterZ,
      17.825 + 17.0 + TR1CenterZ
  };

} // namespace Geometry

#endif // GEOMETRY_H
