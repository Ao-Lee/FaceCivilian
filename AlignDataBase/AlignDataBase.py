from align import AlignDatabase, GetAlignFuncByBoundingBox

if __name__ == '__main__':
    
    source = 'F:\\tmp\\img'
    target = 'F:\\tmp\\aligned'

    F = GetAlignFuncByBoundingBox(output_size=224, margin=24)
    AlignDatabase(source, target, align_func=F)
    
