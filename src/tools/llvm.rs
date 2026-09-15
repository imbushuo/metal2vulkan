//! The small, stable part of LLVM's C API needed for AIR I/O. Each call owns its context.

use libloading::Library;
use std::ffi::{c_char, c_int, c_void, CStr, OsStr};
use std::path::PathBuf;
use std::ptr;
use std::sync::OnceLock;

type Handle = *mut c_void;

macro_rules! llvm_api {
    ($($field:ident: $symbol:ident($($arg:ty),*) -> $result:ty;)*) => {
        struct Llvm {
            $($field: unsafe extern "C" fn($($arg),*) -> $result,)*
            // The function pointers must never outlive their library.
            _library: Library,
        }

        impl Llvm {
            #[cfg(unix)]
            fn cache_fingerprint(&self) -> Result<String, String> {
                use sha2::{Digest, Sha256};
                use std::io::Read;
                let mut info = std::mem::MaybeUninit::<libc::Dl_info>::uninit();
                let found = unsafe {
                    libc::dladdr(self.context_create as *const c_void, info.as_mut_ptr())
                };
                if found == 0 {
                    return Err("cannot identify the loaded LLVM image".into());
                }
                let info = unsafe { info.assume_init() };
                if info.dli_fname.is_null() {
                    return Err("loaded LLVM image has no path".into());
                }
                let path = unsafe { CStr::from_ptr(info.dli_fname) };
                use std::os::unix::ffi::OsStrExt;
                let mut file = std::fs::File::open(OsStr::from_bytes(path.to_bytes()))
                    .map_err(|error| format!("open loaded LLVM image: {error}"))?;
                let mut hash = Sha256::new();
                let mut buffer = [0u8; 65536];
                loop {
                    let count = file.read(&mut buffer)
                        .map_err(|error| format!("hash loaded LLVM image: {error}"))?;
                    if count == 0 { break; }
                    hash.update(&buffer[..count]);
                }
                Ok(format!("{:x}", hash.finalize()))
            }

            #[cfg(not(unix))]
            fn cache_fingerprint(&self) -> Result<String, String> {
                Err("loaded LLVM image identity is unavailable on this platform".into())
            }

            fn open(path: &OsStr) -> Result<Self, String> {
                // Only documented LLVM C ABI symbols are loaded, from the caller-selected library.
                unsafe {
                    let library = Library::new(path).map_err(|error| error.to_string())?;
                    Ok(Self {
                        $($field: *library
                            .get(concat!(stringify!($symbol), "\0").as_bytes())
                            .map_err(|error| error.to_string())?,)*
                        _library: library,
                    })
                }
            }
        }
    };
}

pub(super) fn cache_fingerprint() -> Result<&'static str, String> {
    static IDENTITY: OnceLock<Result<String, String>> = OnceLock::new();
    IDENTITY
        .get_or_init(|| llvm()?.cache_fingerprint())
        .as_deref()
        .map_err(Clone::clone)
}

llvm_api! {
    context_create: LLVMContextCreate() -> Handle;
    context_dispose: LLVMContextDispose(Handle) -> ();
    buffer_create: LLVMCreateMemoryBufferWithMemoryRangeCopy(*const c_char, usize, *const c_char) -> Handle;
    buffer_dispose: LLVMDisposeMemoryBuffer(Handle) -> ();
    buffer_start: LLVMGetBufferStart(Handle) -> *const c_char;
    buffer_size: LLVMGetBufferSize(Handle) -> usize;
    parse_bitcode: LLVMParseBitcodeInContext(Handle, Handle, *mut Handle, *mut *mut c_char) -> c_int;
    parse_ir: LLVMParseIRInContext(Handle, Handle, *mut Handle, *mut *mut c_char) -> c_int;
    verify: LLVMVerifyModule(Handle, c_int, *mut *mut c_char) -> c_int;
    print: LLVMPrintModuleToString(Handle) -> *mut c_char;
    write_bitcode: LLVMWriteBitcodeToMemoryBuffer(Handle) -> Handle;
    module_dispose: LLVMDisposeModule(Handle) -> ();
    message_dispose: LLVMDisposeMessage(*mut c_char) -> ();
}

fn llvm() -> Result<&'static Llvm, String> {
    static LLVM: OnceLock<Result<Llvm, String>> = OnceLock::new();
    LLVM.get_or_init(|| {
        if let Some(path) = crate::env_vars::llvm_library() {
            return Llvm::open(&path)
                .map_err(|error| format!("load LLVM library {}: {error}", PathBuf::from(path).display()));
        }
        let mut errors = Vec::new();
        for path in library_candidates() {
            match Llvm::open(path.as_os_str()) {
                Ok(api) => return Ok(api),
                Err(error) => errors.push(format!("{}: {error}", path.display())),
            }
        }
        Err(format!(
            "cannot load libLLVM; install LLVM or set METAL2VULKAN_LLVM_LIBRARY to its shared library:\n{}",
            errors.join("\n")
        ))
    })
    .as_ref()
    .map_err(Clone::clone)
}

fn library_candidates() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if cfg!(target_os = "macos") {
        for prefix in ["/opt/homebrew/opt", "/usr/local/opt"] {
            paths.push(PathBuf::from(format!("{prefix}/llvm/lib/libLLVM.dylib")));
            for version in (15..=23).rev() {
                paths.push(PathBuf::from(format!(
                    "{prefix}/llvm@{version}/lib/libLLVM.dylib"
                )));
            }
        }
        paths.push(PathBuf::from("libLLVM.dylib"));
    } else {
        paths.push(PathBuf::from("libLLVM.so"));
        for version in (15..=23).rev() {
            paths.push(PathBuf::from(format!("libLLVM-{version}.so")));
            paths.push(PathBuf::from(format!("libLLVM-{version}.so.1")));
            paths.push(PathBuf::from(format!(
                "/usr/lib/llvm-{version}/lib/libLLVM.so"
            )));
        }
    }
    paths
}

struct Context<'a> {
    api: &'a Llvm,
    handle: Handle,
}

impl Drop for Context<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.context_dispose)(self.handle) };
    }
}

struct Module<'a> {
    context: &'a Context<'a>,
    handle: Handle,
}

impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { (self.context.api.module_dispose)(self.handle) };
    }
}

struct Buffer<'a> {
    api: &'a Llvm,
    handle: Handle,
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.buffer_dispose)(self.handle) };
    }
}

impl Llvm {
    fn context(&self) -> Result<Context<'_>, String> {
        let handle = unsafe { (self.context_create)() };
        if handle.is_null() {
            return Err("LLVMContextCreate returned null".into());
        }
        Ok(Context { api: self, handle })
    }

    fn buffer(&self, bytes: &[u8]) -> Result<Buffer<'_>, String> {
        let handle = unsafe {
            (self.buffer_create)(bytes.as_ptr().cast(), bytes.len(), c"metal2vulkan".as_ptr())
        };
        if handle.is_null() {
            return Err("LLVMCreateMemoryBufferWithMemoryRangeCopy returned null".into());
        }
        Ok(Buffer { api: self, handle })
    }

    // All messages come from LLVM, and must be released with the matching library's allocator.
    unsafe fn message(&self, message: *mut c_char) -> String {
        if message.is_null() {
            return "LLVM supplied no diagnostic".into();
        }
        let text = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        unsafe { (self.message_dispose)(message) };
        text
    }
}

pub(super) fn disassemble(bytes: &[u8]) -> Result<String, String> {
    let api = llvm()?;
    let context = api.context()?;
    let buffer = api.buffer(bytes)?;
    let mut module = ptr::null_mut();
    let mut error = ptr::null_mut();
    // Unlike LLVMParseBitcodeInContext2, this entry point returns parse errors instead of
    // sending them to LLVM's default diagnostic handler, which can terminate the process.
    let failed =
        unsafe { (api.parse_bitcode)(context.handle, buffer.handle, &mut module, &mut error) };
    if failed != 0 {
        return Err(format!("LLVM bitcode parse failed:\n{}", unsafe {
            api.message(error)
        }));
    }
    let module = Module {
        context: &context,
        handle: module,
    };
    let text = unsafe { (api.print)(module.handle) };
    if text.is_null() {
        return Err("LLVMPrintModuleToString returned null".into());
    }
    Ok(unsafe { api.message(text) })
}

pub(super) fn assemble(text: &str) -> Result<Vec<u8>, String> {
    let api = llvm()?;
    let context = api.context()?;
    let buffer = api.buffer(text.as_bytes())?;
    // LLVMParseIRInContext consumes the memory buffer on both success and failure.
    let handle = buffer.handle;
    std::mem::forget(buffer);
    let mut module = ptr::null_mut();
    let mut error = ptr::null_mut();
    let failed = unsafe { (api.parse_ir)(context.handle, handle, &mut module, &mut error) };
    if failed != 0 {
        return Err(format!("LLVM IR parse failed:\n{}", unsafe {
            api.message(error)
        }));
    }
    let module = Module {
        context: &context,
        handle: module,
    };
    // LLVMReturnStatusAction = 2: match llvm-as verification without LLVMAbortProcessAction.
    let invalid = unsafe { (api.verify)(module.handle, 2, &mut error) };
    if invalid != 0 {
        return Err(format!("LLVM IR verification failed:\n{}", unsafe {
            api.message(error)
        }));
    }
    if !error.is_null() {
        unsafe { (api.message_dispose)(error) };
    }
    let handle = unsafe { (api.write_bitcode)(module.handle) };
    if handle.is_null() {
        return Err("LLVMWriteBitcodeToMemoryBuffer returned null".into());
    }
    let buffer = Buffer { api, handle };
    let bytes = unsafe {
        std::slice::from_raw_parts(
            (api.buffer_start)(buffer.handle).cast(),
            (api.buffer_size)(buffer.handle),
        )
    };
    Ok(bytes.to_vec())
}
