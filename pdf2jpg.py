"""
PDF转JPG转换工具

将PDF文件的每一页转换为单独的JPG图像文件。

使用PyMuPDF (fitz)库，无需额外安装系统依赖。 

pip install pymupdf
"""

import os
from pathlib import Path
from typing import Union, Optional, List

import fitz  # PyMuPDF


def pdf_to_jpg(
    pdf_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    dpi: int = 300,
    output_prefix: Optional[str] = None,
    fmt: str = "jpg",
    quality: int = 95,
) -> List[Path]:
    """
    将PDF文件转换为JPG图像。
    
    对于多页PDF，每页保存为单独的JPG文件，文件名包含页码序号。
    
    Args:
        pdf_path: PDF文件路径
        output_dir: 输出目录，默认为PDF文件所在目录
        dpi: 输出图像DPI，默认300
        output_prefix: 输出文件名前缀，默认使用PDF文件名
        fmt: 输出格式，默认"jpg"（支持: jpg, png, ppm, pbm）
        quality: JPG质量，1-100，默认95
        
    Returns:
        生成的图像文件路径列表
        
    Example:
        >>> # 单页PDF
        >>> pdf_to_jpg("document.pdf")
        [Path("document_01.jpg")]
        
        >>> # 多页PDF
        >>> pdf_to_jpg("multi_page.pdf", output_dir="./output")
        [Path("output/multi_page_01.jpg"), Path("output/multi_page_02.jpg"), ...]
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError(f"输入文件不是PDF格式: {pdf_path}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = pdf_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定输出文件名前缀
    if output_prefix is None:
        output_prefix = pdf_path.stem
    
    # 打开PDF文档
    print(f"正在转换PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    
    total_pages = len(doc)
    print(f"共 {total_pages} 页")
    
    # 计算序号位数（用于补零对齐）
    num_digits = len(str(total_pages))
    
    # 计算缩放因子 (DPI/72，因为PDF默认72 DPI)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    
    output_paths = []
    for page_num in range(total_pages):
        page = doc[page_num]
        
        # 渲染页面为图像
        pix = page.get_pixmap(matrix=matrix)
        
        # 生成带序号的文件名，如: document_01.jpg, document_02.jpg
        page_str = str(page_num + 1).zfill(num_digits)
        output_filename = f"{output_prefix}_{page_str}.{fmt}"
        output_path = output_dir / output_filename
        
        # 保存图像
        if fmt.lower() in ("jpg", "jpeg"):
            # PyMuPDF保存JPEG需要先转为PNG再用PIL
            from PIL import Image
            import io
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            # 转换为RGB（移除alpha通道，如果有的话）
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(output_path, "JPEG", quality=quality)
        elif fmt.lower() == "png":
            pix.save(str(output_path))
        else:
            # 其他格式直接保存
            pix.save(str(output_path))
        
        print(f"  已保存: {output_path}")
        output_paths.append(output_path)
    
    doc.close()
    print(f"转换完成! 共生成 {len(output_paths)} 个文件")
    return output_paths


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="将PDF文件转换为JPG图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python pdf2jpg.py input.pdf
  python pdf2jpg.py input.pdf -o ./output
  python pdf2jpg.py input.pdf --dpi 200 --quality 90
  python pdf2jpg.py input.pdf --format png
        """
    )
    parser.add_argument("pdf_path", type=str, help="输入的PDF文件路径")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="输出目录（默认为PDF所在目录）")
    parser.add_argument("--dpi", type=int, default=300,
                        help="输出图像DPI（默认300）")
    parser.add_argument("--prefix", type=str, default=None,
                        help="输出文件名前缀（默认使用PDF文件名）")
    parser.add_argument("--format", type=str, default="jpg",
                        choices=["jpg", "jpeg", "png"],
                        help="输出图像格式（默认jpg）")
    parser.add_argument("--quality", type=int, default=95,
                        help="JPG质量，1-100（默认95）")
    
    args = parser.parse_args()
    
    pdf_to_jpg(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        dpi=args.dpi,
        output_prefix=args.prefix,
        fmt=args.format,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()
