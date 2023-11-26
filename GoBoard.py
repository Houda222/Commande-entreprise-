from processing import *


class GoBoard:

    def __init__(self, model, frame):

        self.model = model
        self.frame = frame
        self.results = self.model(frame)
        self.tranformed_image = None
        self.black_stones = []
        self.white_stones = []
        self.annotated_frame = self.results[0].plot(labels=False, conf=False)

    
    def process_frame(self, frame):
        
        self.annotated_frame = self.results[0].plot(labels=False, conf=False)
        # imshow_(self.annotated_frame)
        
        input_points = get_corners(self.results)

        output_edge = 600
        output_points = np.array([[0, 0], [output_edge, 0], [output_edge, output_edge], [0, output_edge]], dtype=np.float32)

        perspective_matrix = cv2.getPerspectiveTransform(input_points, output_points)
        self.transformed_image = cv2.warpPerspective(frame, perspective_matrix, (output_edge, output_edge))
        
        vertical_lines, horizontal_lines = lines_detection(self.results, perspective_matrix)
        
        vertical_lines = removeDuplicates(vertical_lines)
        horizontal_lines = removeDuplicates(horizontal_lines)
        
        vertical_lines = restore_missing_lines(vertical_lines)
        horizontal_lines = restore_missing_lines(horizontal_lines)

        vertical_lines = add_lines_in_the_edges(vertical_lines, "vertical")
        horizontal_lines = add_lines_in_the_edges(horizontal_lines, "horizontal")
        
        vertical_lines = removeDuplicates(vertical_lines)
        horizontal_lines = removeDuplicates(horizontal_lines)
        
        black_stones = get_key_points(self.results, 0, perspective_matrix)
        white_stones = get_key_points(self.results, 6, perspective_matrix)

        cluster_1 = vertical_lines[(vertical_lines<=600).all(axis=1) & (vertical_lines>=0).all(axis=1)]
        cluster_2 = horizontal_lines[(horizontal_lines<=600).all(axis=1) & (horizontal_lines>=0).all(axis=1)]
                
        # img = np.copy(self.transformed_image)
        draw_lines(cluster_1, self.transformed_image)
        # img = np.copy(self.transformed_image)
        draw_lines(cluster_2, self.transformed_image)
        # imshow_(img)
        
        intersections = detect_intersections(cluster_1, cluster_2, self.transformed_image)
                
        if len(intersections) == 0:
            raise Exception(">>>>>No intersection were found!")
        
        return black_stones, white_stones, intersections

    def assign_stones(self, white_stones_transf, black_stones_transf, transformed_intersections):

        self.map = create_board(transformed_intersections)
        
        for stone in white_stones_transf:
            
            cv2.circle(self.transformed_image, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
            
            nearest_corner = None
            closest_distance = 100000
            for inter in transformed_intersections:
                distance = math.dist(inter, stone)
                if distance < closest_distance:
                    nearest_corner = tuple(inter)
                    closest_distance = distance
            # cv2.circle(self.transformed_image, np.array(nearest_corner).astype(dtype=np.int32), 3, (0, 255, 0), 2)
            # print("W", stone, self.map[nearest_corner])
            self.white_stones.append(self.map[nearest_corner][1], self.map[nearest_corner][0]])
            cv2.line(self.transformed_image, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
            # cv2.putText(self.transformed_image, f"{(self.map[nearest_corner])}", nearest_corner, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL , fontScale=0.5, color=(0,0,255))
            
                
        for stone in black_stones_transf:
            
            cv2.circle(self.transformed_image, np.array(stone).astype(dtype=np.int32), 3, (0, 0, 255), 2)
            
            nearest_corner = None
            closest_distance = 100000
            for inter in transformed_intersections:
                distance = math.dist(inter, stone)
                if distance < closest_distance:
                    nearest_corner = tuple(inter)
                    closest_distance = distance
            # cv2.circle(self.transformed_image, np.array(nearest_corner).astype(dtype=np.int32), 3, (0, 255, 0), 2)
            # print("B", stone, self.map[nearest_corner])
            self.black_stones.append(self.map[nearest_corner][1], self.map[nearest_corner][0]])
            cv2.line(self.transformed_image, (int(stone[0]), int(stone[1])), nearest_corner, (0, 255, 255), 2)
            # cv2.putText(self.transformed_image, f"{(self.map[nearest_corner])}", nearest_corner, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL , fontScale=0.5, color=(0,0,255))
        
        # imshow_(self.transformed_image)



