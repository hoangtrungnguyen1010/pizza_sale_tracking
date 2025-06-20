class PersonTracker:
    """Tracks multiple people across frames"""
    
    def __init__(self, max_distance_threshold: int = 150):
        self.people: List[Person] = []
        self.next_id = 1
        self.max_distance_threshold = max_distance_threshold
    @staticmethod
    def update_oven_bbox(new_bbox: Tuple[int, int, int, int]):
        """Update the global oven bounding box for all Person objects"""
        Person.oven_bbox = new_bbox

    def update(self, detections: List[Tuple[int, int, int, int, float]]) -> List[Person]:
        """Update tracker with new detections"""
        new_centers = [((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2, _ in detections]
        
        # Match existing people with new detections
        matched_people = []
        used_detections = set()
        
        for person in self.people:
            best_match = None
            min_distance = float('inf')
            best_idx = -1
            
            for i, (x1, y1, x2, y2, conf) in enumerate(detections):
                if i in used_detections:
                    continue
                
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                distance = person.distance_to(center)
                
                if distance < min_distance and distance < self.max_distance_threshold:
                    min_distance = distance
                    best_match = (x1, y1, x2, y2, conf)
                    best_idx = i
            
            if best_match is not None:
                person.update_position(*best_match)
                matched_people.append(person)
                used_detections.add(best_idx)
        
        # Create new people for unmatched detections
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            if i not in used_detections:
                new_person = Person(
                    id=self.next_id,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    position_history=[]
                )
                matched_people.append(new_person)
                self.next_id += 1
        
        self.people = matched_people
        return self.people
