use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use std::convert::TryFrom;
use vecmat;
// use std::convert::TryInto;
use vecmat::traits::Dot;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

type Point = vecmat::Vector<f64,2>;
type Vector = Point;

type VertexRc = Rc<RefCell<Vertex>>;
type VertexWeak = Weak<RefCell<Vertex>>;

type EdgeRc = Rc<RefCell<Edge>>;
type EdgeWeak = Weak<RefCell<Edge>>;

#[derive(Debug)]
struct Glyph
{
    sym: u16,
    left: i8,
    right: i8,
    strokes: Vec<Vec<(i8,i8)>>
}

#[derive(Debug)]
struct Graph
{
    vertices: Vec<VertexRc>,
    edges: Vec<EdgeRc>
}

fn same_dir(v1: Vector, v2: Vector, cos_min: f64) -> bool
{
    
    v1.dot(v2) >= v1.length() * v2.length() * cos_min
}

impl Graph {
    fn new() -> Graph
    {
        Graph{vertices: Vec::new(), edges: Vec::new()}
    }
    
    fn vertex_for_point(&mut self, p: Point) -> VertexRc
    {
        let i = match self.vertices.iter().enumerate().find(|(_,v)| {v.borrow().point == p}) {
            Some((i,_)) => i,
            None => {
                let v = Rc::new(RefCell::new(
                    Vertex{point: p, edges: Vec::new()}));
                self.vertices.push(v);
                self.vertices.len() - 1
            }
        };
        self.vertices[i].clone()
    }

    fn add_edge(&mut self, l1_vert: VertexRc, l2_vert: VertexRc)
    {
        let (d, len) = {
            let l1 = l1_vert.borrow().point;
            let l2 = l2_vert.borrow().point;
            let d = l2 - l1;
            (d, d.length())
        };
        let new_edge = Rc::new(RefCell::new(
            Edge{end_points: [EndPoint{vertex: Rc::downgrade(&l1_vert),
                                       control: d},
                              EndPoint{vertex: Rc::downgrade(&l2_vert),
                                       control: -d}],
                 len,
                 approximated: Vec::new()}));
        l1_vert.borrow_mut().edges.push(Rc::downgrade(&new_edge));
        l2_vert.borrow_mut().edges.push(Rc::downgrade(&new_edge));
        self.edges.push(new_edge);
    }
/*
    fn get_vertex_edges_mut(&mut self, vertex_index: usize) -> Vec<&mut Edge>
    {
        let mut edges = Vec::new();
        let vertex = &mut self.vertices[vertex_index];
        vertex.edges.sort();
        println!("v_edges: {:?}", vertex.edges);
        let mut edge_slice =  self.edges.as_mut_slice();
        for e in vertex.edges.iter().rev() {
            let (head, tail) = edge_slice.split_at_mut(*e);
            println!("e: {} Head: {}, Tail: {}",e, head.len(),tail.len());
            edges.push(tail.first_mut().unwrap());
            edge_slice = head;
        }
        edges
    }
*/
    fn from_strokes(strokes: &Vec<Vec<(i8,i8)>>) -> Graph
    {
        let mut graph = Graph::new();
        for stroke in strokes.iter() {
            let mut points = stroke.iter();
            if let Some(l1) = points.next() {
                let mut l1_vert = graph.vertex_for_point(
                    Point::from([f64::from(l1.0),f64::from(l1.1)]));
                for l2 in points {
                    let l2_vert = graph.vertex_for_point(
                        Point::from([f64::from(l2.0),f64::from(l2.1)]));
                    graph.add_edge(l1_vert, l2_vert.clone());
                    l1_vert = l2_vert;
                }
            }
        }
        graph
    }
    /*
    fn remove_vertex(&mut self, v_index: usize)
    {
        let v_edges: [usize;2];
        {
            let v = &self.vertices[v_index];
            assert_eq!(v.edges.len(), 2);
            v_edges = v.edges[0..2].try_into().unwrap();
        };
        let edge0 = &self.edges[v_edges[0]];
        let edge1 = &self.edges[v_edges[1]];

        let ep0 = edge0.get_other_end_point(v_index);
        let ep1 = edge1.get_other_end_point(v_index);
        let mut new_edge = Edge{
            len: edge0.len + edge1.len,
            end_points: [ep0.clone(), ep1.clone()],
            approximated: vec![v_index]
        };
        new_edge.approximated.append(&mut edge0.approximated.clone());
        new_edge.approximated.append(&mut edge1.approximated.clone());
        for e in self.vertices[ep1.vertex].edges.iter_mut() {
            if *e == v_edges[1] {
                *e = v_edges[0];
                break
            }
        }
        self.edges[self.vertices[v_index].edges[0]] = new_edge;
        
        self.edges[self.vertices[v_index].edges[1]].len = 0.0;
        self.edges[self.vertices[v_index].edges[1]].approximated = Vec::new();
        self.vertices[v_index].edges.clear();;
    }

    
    pub fn smooth(&mut self)
    {
        for iv in 0..self.vertices.len() {
            if self.vertices[iv].edges.len() == 2 {
                println!("v_edges: {:?}", self.vertices[iv].edges);
                let v_edges = self.get_vertex_edges_mut(iv);

                if same_dir(-v_edges[0].get_this_end_point(iv).control,
                            v_edges[1].get_this_end_point(iv).control,
                            0.8) {
                    if v_edges[0].len <= MAX_SMOOTH_LEN 
                        && v_edges[1].len <= MAX_SMOOTH_LEN
                    {
                        self.remove_vertex(iv);
                    } else if (v_edges[0].len <= MAX_SMOOTH_LEN
                               && !v_edges[1].approximated.is_empty())
                        || (v_edges[1].len <= MAX_SMOOTH_LEN
                            && !v_edges[0].approximated.is_empty())
                    {
                        self.remove_vertex(iv);
                    } else if v_edges[0].len <= MAX_SMOOTH_LEN {
                    }
                }
            }
        }
    }
    */
    pub fn to_svg(&self) -> String {
        let mut buffer = String::from("<path d=\"");
        for e in self.edges.iter() {
            let e = e.borrow();
            if e.len > 0.0 {
                let vert0 = e.end_points[0].vertex.upgrade().unwrap();
                let v0 = &vert0.borrow().point;
                buffer += &format!("M{},{} ",v0.x(),v0.y());
                let vert1 = e.end_points[1].vertex.upgrade().unwrap();
                let v1 = &vert1.borrow().point;
                let c0 = e.end_points[0].control + *v0;
                let c1 = e.end_points[1].control + *v1;
                buffer += &format!("C{},{} {},{} {},{} ",
                                   c0.x(), c0.y(),
                                   c1.x(), c1.y(),
                                   v1.x(),v1.y());
            }
        }
        buffer += "\"/>";
        for e in self.edges.iter() {
            let e = e.borrow();
            for a in e.approximated.iter() {
                let vert = a.borrow();
                buffer += &format!("<circle cx=\"{}\" cy=\"{}\" r=\"1\"/>",
                                   vert.point.x(),
                                   vert.point.y());
            }
        }
        
        buffer
    }
}

#[derive(Debug, Clone)]
struct EndPoint
{
    vertex: Weak<RefCell<Vertex>>,
    control: Vector
}

#[derive(Debug)]
struct Edge
{
    end_points: [EndPoint;2],
    len: f64,
    approximated: Vec<VertexRc>
}

impl Edge
{
    pub fn end_points(&self, vertex: Weak<RefCell<Vertex>>) -> (&EndPoint, &EndPoint)
    {
        if vertex.ptr_eq(&self.end_points[0].vertex) {
            (&self.end_points[0], &self.end_points[1])
        } else {
            (&self.end_points[1], &self.end_points[0])
        }
    }

    pub fn end_points_mut<'a>(&'a mut self, vertex: Weak<RefCell<Vertex>>) -> (&'a mut EndPoint, &'a mut EndPoint)
    {
        if vertex.ptr_eq(&self.end_points[0].vertex) {
            let (head,tail) = self.end_points.split_at_mut(1);
            (&mut head[0], &mut tail[0])
        } else {
            let (head,tail) = self.end_points.split_at_mut(1);
            (&mut tail[0], &mut head[0])
        }
    }

    
    
}


#[derive(Debug)]
struct Vertex
{
    point: Point,
    edges: Vec<EdgeWeak>
}

impl Glyph
{
    pub fn to_svg(&self) -> String {
        let mut buffer = String::from("<path d=\"");
        for stroke in &self.strokes {
            let mut coords = stroke.iter();
            if let Some((x,y)) = coords.next() {
                buffer += &format!("M{},{} ",x,y);
                for (x,y) in coords {
                    buffer += &format!("L{},{} ",x,y);
                }
            }
        }
        buffer += "\"/>";
        buffer
    }
}
fn swap_order<'a, T:PartialOrd>(a: &'a T, b: &'a T) -> (&'a T,&'a T)
{
    if a <= b {
        (a,b)
    } else {
        (b,a)
    }
}
    /*
fn point_splits_line(p: Point, l1: Point, l2: Point) -> bool
{
    let d1 = l2-l1;
    let d2 = p-l1;
    let s = d1.x() * d2.y() - d1.y() * d2.x();
    if s != 0 {return false}
    let (&xmin, &xmax) = swap_order(&l1.x(), &l2.x());
    let (&ymin, &ymax) = swap_order(&l1.y(), &l2.y());
    (xmin+1..xmax).contains(&p.x()) && (ymin+1..ymax).contains(&p.y())
}

fn point_from_i8(p: &(i8,i8)) -> Point
{
    Point::from([i32::from(p.0), i32::from(p.1)])
}



fn split_lines(strokes: &Vec<Vec<(i8,i8)>>)
{
    for stroke1 in strokes.iter() {
        for point in &[stroke1.first(), stroke1.last()] {
            if let Some(end_point) = point {
                for stroke2 in strokes.iter() {
                    let mut points = stroke2.iter();
                    if let Some(mut l1) = points.next() {
                        for l2 in points {
                            let split = point_splits_line(
                                point_from_i8(end_point),
                                point_from_i8(l1),
                                point_from_i8(l2));
                            if split {
                                println!("Split at ({},{}) in (({},{}) - ({},{}))", end_point.0, end_point.1, l1.0, l1.1, l2.0, l2.1);
                            }
                            l1 = l2;
                        }
                    }
                }
            }
        }
    }
}
*/

fn glyphs_to_svg(glyphs: &[Glyph], graphs: &[Graph]) -> String
{
    let mut buffer = String::from("<svg>");
    let mut x_offset = 0i32;
    let y_offset = 0i32;
    for (_index, glyph) in glyphs.iter().enumerate() {
         buffer += &format!("<g transform=\"translate({},{})\" fill=\"none\" stroke=\"black\">\n", 
                            x_offset - i32::from(glyph.left), y_offset);
        buffer += &glyph.to_svg();
        buffer += &format!("<g transform=\"translate({},{})\" fill=\"none\" stroke=\"black\">\n", 0, 50);
        //buffer += &graphs[index].to_svg();
        buffer += "\n</g>\n";
        buffer += "\n</g>\n";
        x_offset += i32::from(glyph.right - glyph.left);
        
    }
    buffer += "</svg>";
    buffer
}


fn coord_from_char(ch: char) -> i8
{
    i8::try_from(ch as i32 - 'R' as i32).unwrap()
}

const MAX_SMOOTH_LEN: f64 = 6.0;

/*
fn smooth(glyph: &Glyph)
{
    for stroke in &glyph.strokes {
        println!("Stroke: {:?}", stroke);
        let mut prev: [Option<(f64,f64)>;2] = [None,None];
        for (x,y) in stroke {
            let (fx,fy) = (f64::from(*x), f64::from(*y));
            if let Some((px0,py0)) = prev[0] {
                let dx = px0 - fx;
                let dy = py0 - fy;
                let len = (dx*dx + dy*dy).sqrt();
                println!("len: {}", len);
                if let Some((px1,py1)) = prev[1] {
                    let dx0 = px1-px0;
                    let dy0 = py1-py0;
                    let c = (dx*dx0 + dy*dy0) / (len * (dx0*dx0 + dy0*dy0).sqrt());
                    let s = (dx*dy0-dy*dx0) / (len * (dx0*dx0 + dy0*dy0).sqrt());

                    println!("cos: {} sin: {}", c, s);
                }
            } 
            prev[1] = prev[0];
            prev[0] = Some((fx,fy));
        }
    }
}*/

fn main() {
    let mut args = env::args();
    args.next();
    let filename = match args.next() {
        Some(f) => f,
        None => {
            eprintln!("Missing font file name");
            return;
        }
    };
    let file = match File::open(&filename) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Failed to open file {}: {}", filename, e);
            return;
        }
    };
    let mut glyphs = Vec::new();
    let mut buffer = BufReader::new(file);
    loop {
        let mut line = String::new();
        match buffer.read_line(&mut line) {
            Ok(0) => {break},
            Ok(_) => {},
            Err(e) => {
                eprintln!("Failed to read file {}: {}", filename, e);
                return;
            }
        };
        let sym: u16 = match line[0..5].trim().parse() {
            Err(e) => {
                eprintln!("Failed to to parse symbol number: {}", e);
                return;
            }
            Ok(s) => s
        };
        let n_verts :u16 = match line[5..8].trim().parse() {
            Err(e) => {
                eprintln!("Failed to to parse number of vertices: {}", e);
                return;
                }
            Ok(s) => s
        };
        let mut pos = line[8..].chars();
        let left = coord_from_char(pos.next().unwrap());
        let right = coord_from_char(pos.next().unwrap());
        let mut strokes = Vec::new();
        let mut stroke: Option<Vec<(i8,i8)>> = None;
        for _ in 1..n_verts {
            let x = coord_from_char(pos.next().unwrap());
            let y = coord_from_char(pos.next().unwrap());
            if x == -50 && y == 0 {
                if let Some(stroke) = stroke.take() {
                    strokes.push(stroke);
                }
            } else {
                if let Some(stroke) = &mut stroke {
                    stroke.push((x,y));
                } else {
                    stroke = Some(vec!((x,y)));
                }
            }
        }
        if let Some(stroke) = stroke.take() {
            strokes.push(stroke);
        }
        let glyph = Glyph {
            sym,
            left,
            right,
            strokes
        };
        glyphs.push(glyph);
    }
    let mut graphs = Vec::new();
    for glyph in &glyphs {
        let mut graph = Graph::from_strokes(&glyph.strokes);
        //graph.smooth();
        graphs.push(graph);
        //println!("{}", graph.to_svg());
        //split_lines(&glyph.strokes);
        //smooth(glyph);
    }
    println!("{}", glyphs_to_svg(&glyphs, &graphs));
    //eprintln!("{:?}", graphs);
}

#[test]
fn point_splits_line_test()
{
    let p = Point::from([4,3]);
    let mut l1 = Point::from([2,2]);
    let mut l2 = Point::from([8,5]);
    assert_eq!(point_splits_line(p,l1,l2), true);
    std::mem::swap(&mut l1, &mut l2);
    assert_eq!(point_splits_line(p,l1,l2), true);
    let p = l1;
    assert_eq!(point_splits_line(p,l1,l2), false);
    let p = l2;
    assert_eq!(point_splits_line(p,l1,l2), false);
    let p = Point::from([5,4]);
    assert_eq!(point_splits_line(p,l1,l2), false);
    let p = Point::from([10,6]);
    assert_eq!(point_splits_line(p,l1,l2), false);
    let p = Point::from([0,1]);
    assert_eq!(point_splits_line(p,l1,l2), false);
}
