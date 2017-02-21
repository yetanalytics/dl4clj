(ns dl4clj.util-test
  (:require [clojure.test :refer :all]
            [dl4clj.utils :refer :all]))

(deftest util-tests
  (testing "the fns in utils"
    (is (= :foobar (camelize :foo-bar )))
    (is (= :FooBar (camelize :foo-bar true)))
    (is (= "fooBar" (camelize "foo-bar")))
    (is (= "fooBar" (camelize "foo bar")))
    (is (= "FooBar" (camelize "Foo Bar")))
    (is (= "foo-bar" (camel-to-dashed "fooBar")))
    (is (= '([1 0] [2 1] [3 2] [4 3]) (indexed [1 2 3 4])))
    ))
