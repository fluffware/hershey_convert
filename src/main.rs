use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use std::convert::TryFrom;
use vecmat;
// use std::convert::TryInto;
use vecmat::traits::Dot;
use std::cell::{RefMut,RefCell};
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

const MAX_SMOOTH_LEN: f64 = 6.0;

fn same_dir(v1: Vector, v2: Vector, cos_min: f64) -> bool
{
    
    v1.dot(v2) >= v1.length() * v2.length() * cos_min
}

fn unlink_edge_from_vertex(vert: &mut Vertex, edge: &EdgeWeak)
{
    vert.edges.retain(|e| !edge.ptr_eq(e));
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

    fn add_edge(&mut self, l1_vert: &VertexRc, l2_vert: &VertexRc) -> EdgeRc
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
        self.edges.push(new_edge.clone());
        new_edge
    }
/*
    fn get_vertex_edges_mut(&mut self, vertex_index: usize) -> Vec<&mut Edge>
    {
        let mut edges = Vec::new();
        let vertex = &mut self.vertices[vertex_index];
        vertex.edges.sort();
    eprintln!("v_edges: {:?}", vertex.edges);
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
                    graph.add_edge(&l1_vert, &l2_vert.clone());
                    l1_vert = l2_vert;
                }
            }
        }
        graph
    }

    
    fn unlink_vertex(&mut self, vert: &VertexRc)
    {
        assert_eq!(vert.borrow().edges.len(), 2);
        let edge0_rc;
        let edge1_rc;
        {
            let mut vert_mut = vert.borrow_mut();
            
            edge0_rc = vert_mut.edges[0].upgrade().unwrap();   
            edge1_rc = vert_mut.edges[1].upgrade().unwrap();
            vert_mut.edges.clear();
        }
        let vert0_rc;
        let vert1_rc;
        let end_points;
        {
            let mut edge0 = edge0_rc.borrow_mut();
            let mut edge1 = edge1_rc.borrow_mut();
            let vert_weak = VertexRc::downgrade(vert);
            let (_, ep0) = edge0.end_points_mut(&vert_weak);
            let (_, ep1) = edge1.end_points_mut(&vert_weak);
            end_points = [ep0.clone(), ep1.clone()];
            vert0_rc = ep0.vertex.upgrade().unwrap();
            vert1_rc = ep1.vertex.upgrade().unwrap();
        }
        if Rc::ptr_eq(&vert0_rc, &vert1_rc) {return}
        let mut vert0 = vert0_rc.borrow_mut();
        let mut vert1 = vert1_rc.borrow_mut();
        unlink_edge_from_vertex(&mut vert0, &EdgeRc::downgrade(&edge0_rc));
        unlink_edge_from_vertex(&mut vert1, &EdgeRc::downgrade(&edge1_rc));
        self.edges.retain(|e| !(Rc::ptr_eq(e,&edge0_rc) || Rc::ptr_eq(e,&edge1_rc)));
        
        let mut edge0 = Rc::try_unwrap(edge0_rc).unwrap().into_inner();
        let mut edge1 = Rc::try_unwrap(edge1_rc).unwrap().into_inner();
        
        let mut new_edge = Edge{
            len:  edge0.len + edge1.len,
            end_points,
            approximated: vec![VertexRc::downgrade(vert)]
        };
                

        new_edge.approximated.append(&mut edge0.approximated);
        new_edge.approximated.append(&mut edge1.approximated);
        let new_edge_rc = Rc::new(RefCell::new(new_edge));
        let new_edge_weak = EdgeRc::downgrade(&new_edge_rc);
        self.edges.push(new_edge_rc);
        vert0.edges.push(new_edge_weak.clone());
        vert1.edges.push(new_edge_weak);
    

    }

    pub fn smooth(&mut self)
    {
        let mut remove = Vec::new();
        for vert in &self.vertices {
            if vert.borrow().edges.len() == 2 {
                let v_edges_rc: Vec<EdgeRc> = 
                    vert.borrow().edges.iter()
                    .map(|e| Weak::upgrade(e).unwrap()).collect();
                let mut v_edges: Vec<RefMut<Edge>>  =
                    v_edges_rc.iter()
                    .map(|e| e.borrow_mut()).collect();
                let vert_weak = Rc::downgrade(&vert);
                let control0 = v_edges[0].end_points(&vert_weak).0.control;
                let control1 = v_edges[1].end_points(&vert_weak).0.control;
                if same_dir(-control0,
                            control1,
                            0.7) {
                    let e0_smoothed =
                        (v_edges[0].len <= MAX_SMOOTH_LEN
                         || !v_edges[0].approximated.is_empty())
                        && !(control0.x() == 0.0
                             || control0.y() == 0.0);
                        
                    let e1_smoothed =
                        (v_edges[1].len <= MAX_SMOOTH_LEN
                         || !v_edges[1].approximated.is_empty())
                        && !(control1.x() == 0.0
                             || control1.y() == 0.0);
                        
                    
                    if e0_smoothed && e1_smoothed
                    {
                        remove.push(vert.clone());
                    } else if e0_smoothed
                    {
                        v_edges[0].end_points_mut(&vert_weak).0.control = 
                            -control1;
                    } else if e1_smoothed
                    {
                        v_edges[1].end_points_mut(&vert_weak).0.control = 
                            -control0;
                    } 
                }
            }
        }
        for vert in &remove {
            self.unlink_vertex(vert);
        }
    }

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
                if let Some(vert_rc) = Weak::upgrade(a) {
                    let vert = vert_rc.borrow();
                    buffer += &format!("<circle cx=\"{}\" cy=\"{}\" r=\"1\"/>",
                                       vert.point.x(),
                                       vert.point.y());
                }
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
    approximated: Vec<VertexWeak>
}

impl Edge
{
    pub fn end_points(&self, vertex: &VertexWeak) -> (&EndPoint, &EndPoint)
    {
        if vertex.ptr_eq(&self.end_points[0].vertex) {
            (&self.end_points[0], &self.end_points[1])
        } else {
            (&self.end_points[1], &self.end_points[0])
        }
    }

    pub fn end_points_mut<'a>(&'a mut self, vertex: &VertexWeak) -> (&'a mut EndPoint, &'a mut EndPoint)
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

impl Vertex
{
    fn new(x: f64, y: f64) -> Vertex
    {
        Vertex{point: Point::from([x,y]),
               edges: Vec::new()
        }
    }
    
    fn new_rc(x: f64, y: f64) -> VertexRc
    {
        Rc::new(RefCell::new(Self::new(x,y)))
    }
    
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
    for (index, glyph) in glyphs.iter().enumerate() {
         buffer += &format!("<g transform=\"translate({},{})\" fill=\"none\" stroke=\"black\">\n", 
                            x_offset - i32::from(glyph.left), y_offset);
        buffer += &glyph.to_svg();
        buffer += &format!("<g transform=\"translate({},{})\" fill=\"none\" stroke=\"black\">\n", 0, 50);
        buffer += &graphs[index].to_svg();
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
        graph.smooth();
        //println!("{}", graph.to_svg());
        graphs.push(graph);
        //split_lines(&glyph.strokes);
        //smooth(glyph);
    }
    println!("{}", glyphs_to_svg(&glyphs, &graphs));
    //eprintln!("{:?}", graphs);
}

/*
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
*/
#[test]
fn remove_vertex_test()
{
    let mut graph = Graph::new();
    let vert1 = Vertex::new_rc(1.0,5.0);
    let vert2 = Vertex::new_rc(3.0,5.0);
    let vert3 = Vertex::new_rc(7.0,4.0);
    graph.vertices.append(&mut vec![vert1.clone(), vert2.clone(), vert3.clone()]);
    graph.add_edge(&vert1, &vert2);
    graph.add_edge(&vert2, &vert3);
    
    assert_eq!(vert1.borrow().edges.len(),1);
    assert_eq!(vert2.borrow().edges.len(),2);
    assert_eq!(vert3.borrow().edges.len(),1);
    
    assert_eq!(Rc::strong_count(&vert2), 2);

    graph.unlink_vertex(&vert2);
    assert_eq!(Rc::strong_count(&vert1), 2);
    assert_eq!(Rc::strong_count(&vert2), 2);
    assert_eq!(Rc::strong_count(&vert3), 2);

    let edge0_rc = Weak::upgrade(&vert1.borrow().edges[0]).unwrap();
    let edge0 = edge0_rc.borrow();
    
    let (ep1, ep2) = edge0.end_points(&Rc::downgrade(&vert1));

    assert!(Weak::ptr_eq(&ep1.vertex, &Rc::downgrade(&vert1)));
    assert!(Weak::ptr_eq(&ep2.vertex, &Rc::downgrade(&vert3)));

     let (ep1, ep2) = edge0.end_points(&Rc::downgrade(&vert2));

    assert!(Weak::ptr_eq(&ep1.vertex, &Rc::downgrade(&vert3)));
    assert!(Weak::ptr_eq(&ep2.vertex, &Rc::downgrade(&vert1)));

    
}
