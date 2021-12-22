import numpy as np
# point1 = [13.1,34.6,2.4456]
# point2 = [56.44665,34.2234,78.55]
# dist = np.sqrt(np.sum(np.square(a-b)))
# dist = 0
# for x,y in zip(a,b):
# 	dist = dist + np.power(np.absolute((x-y)),3)
# print(float(np.cbrt(dist)))

# a = np.array((1, 2, 3))
# b = np.array((4, 5, 6))

# dist = np.sqrt(np.sum(np.square(a-b)))

# print(dist)
# for i,j in zip(point1,point2):
# 	if np.linalg.norm(i) == 0 or np.linalg.norm(j) == 0:
# 		print("here")
# 		print(float(1))
# 	else:
# 		# print('here1')
# 		# cos_sim = float(np.dot(i, j)/(np.linalg.norm(i)*np.linalg.norm(j)))
# 		print(np.dot(i,j))
# 		print(np.linalg.norm(i)*np.linalg.norm(j))
# 		print()

# print("Done")
# norm_point1 = 0
# norm_point2 = 0
# product_sum = 0
# for x, y in zip(point1, point2):
# 	norm_point1 = norm_point1 + x ** 2
# 	norm_point2 = norm_point2 + y ** 2
# 	product_sum = product_sum + x * y
# 	if norm_point1 == 0 or norm_point2 == 0:
# 		print("1")
# 	# print(float(1 - float(product_sum) / float(np.sqrt(norm_point1) * np.sqrt(norm_point2))))
# 	print(product_sum)
# 	print(np.sqrt(norm_point1) * np.sqrt(norm_point2))
# 	print()

# features = [[2, -1], [-1, 5], [0, 0]]
# total_data = len(features)
# total_features = len(features[0])
# normalized_features = [[0] * total_features for _ in range(total_data)]
# minimum_feature_list = [float("inf")] * total_features
# maximum_feature_list = [float("-inf")] * total_features
# for y in range(total_data):
# 	for x in range(total_features):
# 		val = features[y][x]
# 		maximum_feature_list[x] = max(maximum_feature_list[x], val)
# 		minimum_feature_list[x] = min(minimum_feature_list[x], val)
# for y in range(total_data):
# 	for x in range(total_features):
# 		max_min_diff = maximum_feature_list[x] - minimum_feature_list[x]
# 		print(max_min_diff)
# 		normalized_features[y][x] = 0 if max_min_diff == 0 else ((features[y][x] - minimum_feature_list[x]) / max_min_diff)
# 		print(normalized_features[y][x])
# print (normalized_features)

