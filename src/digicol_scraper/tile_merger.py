"""
Tile Merger module for digicol-scraper.

功能：使用 Pillow 将 DeepZoom 瓦片拼接为完整的超高分辨率图片。
"""

import os
import re
from typing import Dict, Optional, Tuple
from PIL import Image
from tqdm import tqdm


class TileMerger:
    """
    Tile Merger class.

    负责将下载的 DeepZoom 瓦片拼接为完整的图片。
    """

    def __init__(self, tile_dir: str, tile_size: int = 510) -> None:
        """
        Initialize tile merger.

        Args:
            tile_dir: Tile directory path
            tile_size: Tile size (default 510)
        """
        self.tile_dir = tile_dir
        self.tile_size = tile_size

    def _get_level_dir(self, level: int) -> str:
        """
        Get directory path for specified level.

        Args:
            level: Tile level

        Returns:
            Level directory full path
        """
        level_dir_name = f"level_{level}"
        if os.path.basename(self.tile_dir) == level_dir_name:
            return self.tile_dir
        return os.path.join(self.tile_dir, level_dir_name)

    def get_tiles_for_level(self, level: int) -> Dict[Tuple[int, int], str]:
        """
        Get tile dictionary for specified level.

        Args:
            level: Tile level

        Returns:
            Tile dictionary with (col, row) tuple as key and file path as value
            Filename format: col_row.png
        """
        level_dir = self._get_level_dir(level)
        if not os.path.exists(level_dir):
            return {}

        tiles = {}
        for filename in os.listdir(level_dir):
            match = re.match(r"(\d+)_(\d+)\.png", filename)
            if match:
                tiles[(int(match.group(1)), int(match.group(2)))] = os.path.join(
                    level_dir, filename
                )
        return tiles

    def get_image_dimensions(
        self, tiles: Dict[Tuple[int, int], str]
    ) -> Tuple[int, int]:
        """
        Calculate complete image dimensions from edge tiles.

        Args:
            tiles: Tile dictionary

        Returns:
            (width, height) tuple
        """
        if not tiles:
            return 0, 0

        max_col = max(c for c, _ in tiles)
        max_row = max(r for _, r in tiles)

        width = max(
            (
                c * self.tile_size
                + Image.open(tiles[(c, r)]).width
                for c, r in tiles
                if c == max_col and os.path.exists(tiles[(c, r)])
            ),
            default=0,
        )

        height = max(
            (
                r * self.tile_size
                + Image.open(tiles[(c, r)]).height
                for c, r in tiles
                if r == max_row and os.path.exists(tiles[(c, r)])
            ),
            default=0,
        )

        return width, height

    def merge_level(
        self,
        level: int,
        output_path: str,
        tiles: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> bool:
        """
        Merge tiles for specified level.

        Args:
            level: Tile level
            output_path: Output image path
            tiles: Optional tile dictionary, auto-fetched if not provided

        Returns:
            True on success, False on failure
        """
        if tiles is None:
            tiles = self.get_tiles_for_level(level)

        if not tiles:
            return False

        width, height = self.get_image_dimensions(tiles)
        result = Image.new("RGB", (width, height), (255, 255, 255))

        for (col, row), tile_path in tqdm(
            tiles.items(), desc=f"Merging level {level}"
        ):
            try:
                tile = Image.open(tile_path)
                result.paste(tile, (col * self.tile_size, row * self.tile_size))
                tile.close()
            except Exception:
                pass

        result = result.crop((0, 0, width, height))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result.save(output_path, "PNG")
        result.close()
        return True

    def find_highest_level(self) -> int:
        """
        Find highest available tile level.

        Returns:
            Highest level number, -1 if not found
        """
        levels = [
            int(d.replace("level_", ""))
            for d in os.listdir(self.tile_dir)
            if d.startswith("level_")
        ]
        return max(levels) if levels else -1

    def merge_highest_level(self, output_path: str) -> bool:
        """
        Merge highest resolution level tiles.

        Args:
            output_path: Output image path

        Returns:
            True on success, False on failure
        """
        level = self.find_highest_level()
        if level < 0:
            return False
        return self.merge_level(
            level, output_path, self.get_tiles_for_level(level)
        )


def main() -> None:
    """Main function (shows message when no test data)."""
    print("No test tiles directory configured")


if __name__ == "__main__":
    main()
