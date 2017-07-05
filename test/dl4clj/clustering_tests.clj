(ns dl4clj.clustering-tests
  (:require [clojure.test :refer :all]
            [dl4clj.clustering.kmeans-clustering :refer [set-up-kmeans]]
            [dl4clj.clustering.cluster.cluster :refer :all]
            [dl4clj.clustering.cluster.point :refer :all]
            [dl4clj.clustering.cluster.cluster-set :refer :all]
            ;; making my life easier
            [nd4clj.linalg.factory.nd4j :refer [vec-or-matrix->indarray]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; points
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; used in cluster testing
(def center-point (new-point :data [0 0 0 0]
                             :label "origin"))

(def first-point (new-point :data [1 2 3 4]
                            :label "pos"))

(def second-point (new-point :data [-1 -2 -3 -4]
                             :label "neg"))

;; used in point testing

(def all-args-point (new-point :data [5 6 7 8]
                               :label "test point"
                               :id "all-args-point"))

(def set-vals (as-> (new-point) p
                (set-point-data! :point p :data [6 7 8 9])
                (set-point-id! :point p :id "I set this id")
                (set-point-label! :point p :label "I set this label")))

(deftest point-testing
  (testing "the creation and getter/setter fns of points"
    (is (= org.deeplearning4j.clustering.cluster.Point
           (type (new-point))))

    (is (= (vec-or-matrix->indarray [5 6 7 8])
           (get-point-data all-args-point)))

    (is (= "all-args-point"
           (get-point-id all-args-point)))

    (is (= "test point"
           (get-point-label all-args-point)))

    (is (= "I set this label"
           (get-point-label set-vals)))

    (is (= "I set this id"
           (get-point-id set-vals)))

    (is (= (vec-or-matrix->indarray [6 7 8 9])
           (get-point-data set-vals)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; clusters
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; for setting the center of an empty cluster do this
;; but not sure how to then set the distance fn
;; cant get distance to center without a distance-fn

(def start-with-empty-cluster
  (as-> (new-cluster) c
    (set-center! :cluster c :point center-point)
    (add-point! :cluster c :point first-point :move-center? true)
    (add-point! :cluster c :point second-point :move-center? true)))

;; this ^ works, but from what I can tell, you cant add a distance fn at this level
;; better to start off with a cluster w/ a center and distance fn

(def add-multiple-points
  (as-> (new-cluster :center-point center-point
                     :distance-fn "manhattan") c
    ;; could also set each point individually
    (set-points! :cluster c :points (list first-point second-point))))

;; in both cases we have our center

(is (= (get-center start-with-empty-cluster)
       (get-center add-multiple-points)))

;; but our start with empty cluster doesn't have a distance fn attached
;; this matters because we can no figure out the distance to center given a point

(is (= "java.lang.NullPointerException"
       (try (get-distance-to-center :cluster start-with-empty-cluster
                                    :point first-point)
            (catch Exception e (str e)))))
(is (= 10.0 (get-distance-to-center :cluster add-multiple-points
                                    :point first-point)))

;; we are going to be working with our add-multiple-points now

;; naturally we can set the cluster id and label
(def cluster-with-meta
  (as-> add-multiple-points c
    (set-cluster-label! :cluster c :label "test cluster")
    (set-cluster-id! :cluster c :id "cluster id")))


(is (= "test cluster" (get-cluster-label cluster-with-meta)))
(is (= "cluster id" (get-cluster-id cluster-with-meta)))
;; verifying setting cluster meta data doesn't change the points
(is (= (get-point-data first-point)
       (get-point-data (get-point :cluster cluster-with-meta
                                  :point-id (-> cluster-with-meta
                                                get-points
                                                first
                                                get-point-id)))))

;; and we can empty the cluster

(def removed-all-points (remove-points! start-with-empty-cluster))

(is (true? (empty-cluster? removed-all-points)))

;; but it would be better to just create a new cluster

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; cluster sets
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; cluster can be completley empty or start with a distance fn
;; if they start empty, you can add a distance fn

(def add-distance-to-cluster-set (set-accumulation! :cluster-set (new-cluster-set)
                                                    :distance-fn "euclidean"))

(def cluster-set (new-cluster-set :distance-fn "euclidean"))

(is (= (get-accumulation cluster-set)
       (get-accumulation add-distance-to-cluster-set)))

;; two ways to add clusters to a cluster set

;; 1)
;; every time you call add-new-cluster-with-center!
;; a new cluster with the given center is created
;; and added to the cluster set

(is (= 4 (get-cluster-count
          (let [c-s (new-cluster-set :distance-fn "manhattan")]
            (dotimes [n 4]
              (add-new-cluster-with-center!
               :cluster-set c-s
               :center-point center-point))
            ;; do times returns nil
            ;; so need to manually return our cluster set
            c-s))))

;; 2)
;; manually add clusters to the set

(def second-test-cluster
  ;; create a cluster
  (as-> (new-cluster :center-point center-point
                     :distance-fn "manhattan") c
    ;; could also set each point individually
    (set-points! :cluster c :points (list first-point second-point))))

(def another-cluster-with-meta
  ;; set its meta data
  (as-> second-test-cluster c
    (set-cluster-label! :cluster c :label "second test cluster")
    (set-cluster-id! :cluster c :id "second cluster id")))

(is (= 2 (get-cluster-count
          (set-clusters!
           :cluster-set (new-cluster-set :distance-fn "manhattan")
           ;; we just created one of the clusters with meta data
           ;; the other one was created above
           :clusters (list cluster-with-meta another-cluster-with-meta)))))

(def cluster-set-with-labeled-clusters
 (set-clusters!
 :cluster-set (new-cluster-set :distance-fn "manhattan")
 :clusters (list cluster-with-meta another-cluster-with-meta)))

;; lets get that meta data back out

(is (= `("test cluster" "second test cluster")
       (map get-cluster-label (get-clusters cluster-set-with-labeled-clusters))))
(is (= `("cluster id" "second cluster id")
       (map get-cluster-id (get-clusters cluster-set-with-labeled-clusters))))

;; our second way of adding clusters to a cluster set is more desirable as we can
;; set our cluster meta data easier
;; we could also set the meta data for the clusters in our first cluster set
;; but it is a lengthier process

(let [cluster-s (new-cluster-set :distance-fn "euclidean")
      with-clusters (do (dotimes [n 2]
                          (add-new-cluster-with-center!
                           :cluster-set cluster-s
                           :center-point center-point))
                        cluster-s)
      clusters (get-clusters with-clusters)
      [c-1 c-2] clusters]
  ;; notice we are working with stateful java objects
  ;; we dont have to add our updated clusters back into the set
  (as-> c-1 c
    (set-cluster-id! :id "cluster 1" :cluster c)
    (set-cluster-label! :label "foo" :cluster c))
  (as-> c-2 c
    (set-cluster-id! :id "cluster 2" :cluster c)
    (set-cluster-label! :label "baz" :cluster c))
  (is (= `("foo" "baz")
         (map get-cluster-label (get-clusters with-clusters))))
  (is (= `("cluster 1" "cluster 2")
         (map get-cluster-id (get-clusters with-clusters)))))

;; if you don't care about all of your clusters being easily identified
;; thats fine, you normally only care about a single cluster

(let [cluster-s (new-cluster-set :distance-fn "euclidean")
      with-clusters (do (dotimes [n 2]
                          (add-new-cluster-with-center!
                           :cluster-set cluster-s
                           :center-point center-point))
                        cluster-s)
      clusters (get-clusters with-clusters)
      [c-1 c-2] clusters]
  (get-cluster-id c-1))

(dl4clj.berkeley/get-first
 (nerest-cluster-to-point
  :cluster-set cluster-set-with-labeled-clusters
  :point (new-point :data [1 2 3 4])))

(classify-point :point (new-point :data [1 2 3 4])
                :cluster-set (add-new-cluster-with-center!
                              :cluster-set (new-cluster-set :distance-fn "euclidean")
                              :center-point center-point)
                :move-cluster-center? false)

(classify-point :point (new-point :data [1 2 3 4])
                :cluster-set cluster-set-with-labeled-clusters
                :move-cluster-center? true)

(get-point-distribution cluster-set-with-labeled-clusters)

;; points
;;all-args-point
;;set-vals

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; classify points-with-meta in a cluster-with-meta in a cluster set
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;set-points! now works

(let [p1 (new-point :data [-1 -2 -3 -4]
                    :label "p1"
                    :id "p1")
      p2 (new-point :data [1 2 3 4]
                    :label "p2"
                    :id "p2")
      p3 (new-point :data [-2 -4 -6 -8]
                    :label "p3"
                    :id "p3")
      p4 (new-point :data [6 12 18 24]
                    :label "p4"
                    :id "p4")
      c1 (new-point :data [0 0 0 0]
                    :label "origin"
                    :id "c1")
      c2 (new-point :data [2 4 6 8]
                    :label "middle"
                    :id "c2")
      cluster-1 (as-> (new-cluster :center-point c1 :distance-fn "manhattan") c
                  (set-points! :cluster c :points (list p1 p2))
                  (set-cluster-label! :cluster c :label "first cluster")
                  (set-cluster-id! :cluster c :id "c1"))
      cluster-2 (as-> (new-cluster :center-point c2 :distance-fn "manhattan") c
                  (set-points! :cluster c :points (list p3 p4))
                  (set-cluster-label! :cluster c :label "second cluster")
                  (set-cluster-id! :cluster c :id "c2"))
      cs (set-clusters! :cluster-set (new-cluster-set :distance-fn "manhattan")
                        :clusters (list cluster-1 cluster-2))]
  (try (classify-point :point (new-point :data [4 8 12 16])
                         :cluster-set cs
                         :move-cluster-center? false)
         (catch Exception e (str e))))

;; no meta and manualy add-point (no meta and add points still errors)
;; this one works
(let [p1 (new-point :data [-1 -2 -3 -4])
      p2 (new-point :data [1 2 3 4])
      p3 (new-point :data [-2 -4 -6 -8])
      p4 (new-point :data [6 12 18 24])
      c1 (new-point :data [0 0 0 0])
      c2 (new-point :data [2 4 6 8])
      cluster-1 (as-> (new-cluster :center-point c1 :distance-fn "manhattan") c
                  (add-point! :cluster c :point p1)
                  (add-point! :cluster c :point p2))

      cluster-2 (as-> (new-cluster :center-point c2 :distance-fn "manhattan") c
                  (add-point! :cluster c :point p3)
                  (add-point! :cluster c :point p4))
      cs (set-clusters! :cluster-set (new-cluster-set :distance-fn "manhattan")
                        :clusters (list cluster-1 cluster-2))]
  (try (classify-point :point (new-point :data [4 8 12 16])
                  :cluster-set cs
                  :move-cluster-center? false)
       (catch Exception e (str e))))

;; now lets try with meta data attached to points and clusters
;; but add-point! used
;; this works

(let [p1 (new-point :data [-1 -2 -3 -4]
                    :label "p1"
                    :id "p1")
      p2 (new-point :data [1 2 3 4]
                    :label "p2"
                    :id "p2")
      p3 (new-point :data [-2 -4 -6 -8]
                    :label "p3"
                    :id "p3")
      p4 (new-point :data [6 12 18 24]
                    :label "p4"
                    :id "p4")
      c1 (new-point :data [0 0 0 0]
                    :label "origin"
                    :id "c1")
      c2 (new-point :data [2 4 6 8]
                    :label "middle"
                    :id "c2")
      cluster-1 (as-> (new-cluster :center-point c1 :distance-fn "manhattan") c
                  (set-cluster-label! :cluster c :label "first cluster")
                  (set-cluster-id! :cluster c :id "c1")
                  (add-point! :cluster c :point p1)
                  (add-point! :cluster c :point p2))
      cluster-2 (as-> (new-cluster :center-point c2 :distance-fn "manhattan") c
                  (set-cluster-label! :cluster c :label "second cluster")
                  (set-cluster-id! :cluster c :id "c2")
                  (add-point! :cluster c :point p3)
                  (add-point! :cluster c :point p4))
      cs (set-clusters! :cluster-set (new-cluster-set :distance-fn "manhattan")
                        :clusters (list cluster-1 cluster-2))]
  (try (classify-point :point (new-point :data [4 8 12 16])
                  :cluster-set cs
                  :move-cluster-center? false)
       (catch Exception e (str e))))




;; cluster set = cluster-set-with-labeled-clusters


(def kmeans-clustering (set-up-kmeans :n-clusters 2
                                      :distance-fn "euclidean"
                                      :max-iterations 2))

(def other-kmeans (set-up-kmeans :n-clusters 2 :min-distribution-variation-rate 0.2
                                 :distance-fn "manhattan" :allow-empty-clusters? true))
