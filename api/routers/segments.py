"""api/routers/segments.py"""

from typing import List

from fastapi import APIRouter, Depends

from api.dependencies import verify_api_key
from api.schemas.customer import SegmentProfile
from api.services.customer_service import get_segment_profiles

router = APIRouter(prefix="/segments", tags=["segments"], dependencies=[Depends(verify_api_key)])


# Phase 3's k=4 segmentation result, one row per segment -- feeds the
# dashboard's segment explorer in Phase 8
@router.get("", response_model=List[SegmentProfile])
def list_segments():
    return get_segment_profiles()