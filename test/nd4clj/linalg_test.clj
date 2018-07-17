(ns nd4clj.linalg-test
  (:require [clojure.test :refer :all]
            [nd4clj.linalg.factory.nd4j :refer [shape tensor->indarray]]))

(deftest shape-test
  (testing "matrix shape"
    (is (= [0] (shape [])))
    (is (= [2] (shape [1 1])))
    (is (= [2 2] (shape [[1 1] [1 1]])))
    (is (= [2 2 1] (shape [[[1] [1]] [[1] [1]]])))
    (is (= [1 2 1 1] (shape [[[[1]] [[1]]]])))
    (is (= [1 2 1 1 1] (shape [[[[[1]]] [[[1]]]]])))
    (is (= [2 2 1 2 1]
           (shape [[[[[1] [2]]] [[[2] [10]]]] [[[[2] [10]]] [[[10] [20]]]]])))))

(deftest tensor->indarray-test
  (testing "tensor->indarray"
    (is (= [0]
           (->> [] tensor->indarray .shape vec next)))
    (is (= [2]
           (->> [1 1] tensor->indarray .shape vec next)))
    (is (= [2 2]
           (->> [[1 1] [1 1]] tensor->indarray .shape vec)))
    (is (= [2 2 1]
           (->> [[[1] [1]] [[1] [1]]] tensor->indarray .shape vec)))
    (is (= [1 2 1 1]
           (->> [[[[1]] [[1]]]] tensor->indarray .shape vec)))))
