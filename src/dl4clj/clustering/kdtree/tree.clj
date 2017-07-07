(ns dl4clj.clustering.kdtree.tree
  (:import [org.deeplearning4j.clustering.kdtree KDTree KDTree$KDNode])
  (:require [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

(defn new-kdtree
  "KDTree based on: https://github.com/nicky-zs/kdtree-python/blob/master/kdtree.py"
  [dims]
  (KDTree. dims))

(defn new-kdnode
  "create a node given a point"
  [point]
  (KDTree$KDNode. (vec-or-matrix->indarray point)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; tree api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn insert-point!
  "inserts a point into the supplied kdtree

  :point (INDarray or vec), the vector to add to the tree"
  [& {:keys [kdtree point]}]
  (doto kdtree (.insert (vec-or-matrix->indarray point))))

(defn delete-point!
  "deletes a point from the node that contains it and returns that node

  :point (INDarray or vec), the point to be removed from the tree"
  [& {:keys [kdtree point]}]
  (.delete kdtree (vec-or-matrix->indarray point)))

(defn knn
  "returns the nearest neighbors within the suplied distance from the supplied point

  :point (INDarray or vec), the point you want neighbors of

  :distance (double), the distance away from the point to search"
  [& {:keys [point distance]}]
  (.knn (vec-or-matrix->indarray point) distance))

(defn get-nearest-neighbor
  "returns the point closest to the supplied point and the distance between them

  point (INDarray or vec), the point you want neighbors of"
  [point]
  (.nn (vec-or-matrix->indarray point)))

(defn get-tree-size
  "returns the size of the supplied tree"
  [kdtree]
  (.size kdtree))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; node api fns
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn get-left
  [kdnode]
  (.getLeft kdnode))

(defn get-right
  [kdnode]
  (.getRight kdnode))

(defn get-parent-node
  [kdnode]
  (.getParent kdnode))

(defn get-point-from-node
  [kdnode]
  (.getPoint kdnode))

(defn set-left!
  [& {:keys [node left-node]}]
  (doto node (.setLeft left-node)))

(defn set-right!
  [& {:keys [node right-node]}]
  (doto node (.setRight right-node)))

(defn set-parent!
  [& {:keys [node parent-node]}]
  (doto node (.setParent parent-node)))
